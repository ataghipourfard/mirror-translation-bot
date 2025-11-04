"""
Discord Translator/Mirroring Bot — Self-Hosted "Premium for me"

Features:
- Channel mirroring with translation (bridge many-to-many)
- Slash commands: /bridge add|remove|list, /premium grant|revoke|status, /lang list
- Premium gating: unlimited bridges & autotranslate for owner guild/user only; others limited
- Optional autotranslate mode for a channel
- Supports multiple translation backends via adapters (DeepL, Google Cloud, Offline MarianMT demo)
- SQLite storage (aiosqlite)
- Basic rate limiting and loop-avoidance

IMPORTANT: Fill in environment variables before running (see bottom of file).
Python 3.10+
"""
import io
import os
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import discord
from discord import app_commands, AllowedMentions
from discord.ext import commands

from dotenv import load_dotenv
load_dotenv()

# Storage
import aiosqlite

# -------------- Logging --------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("heph-selfhost")

# -------------- Config --------------
BOT_TOKEN = os.getenv("DISCORD_TOKEN", "")
OWNER_USER_ID = int(os.getenv("OWNER_USER_ID", "0"))  # Your Discord user ID for premium control
OWNER_GUILD_ID = int(os.getenv("OWNER_GUILD_ID", "0"))  # Your primary server ID (optional)
DB_PATH = os.getenv("DB_PATH", "translator.db")
TRANSLATOR_BACKEND = os.getenv("TRANSLATOR_BACKEND", "deepl").lower()  # deepl|google|offline
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "")  # if using Google Cloud Translate
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "global")
DEFAULT_TARGET_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")

# -------------- Premium Policy --------------
MAX_BRIDGES_FREE = 2  # non-premium guild hard-limit

# -------------- Translator Adapters --------------
class TranslatorAdapter:
    async def translate(self, text: str, target_lang: str) -> str:
        raise NotImplementedError

class DeepLAdapter(TranslatorAdapter):
    def __init__(self, api_key: str):
        try:
            import httpx  # type: ignore
        except ImportError:
            raise RuntimeError("Install httpx for DeepL adapter: pip install httpx")
        self.httpx = __import__("httpx")
        self.api_key = api_key
        self.endpoint = (
            "https://api-free.deepl.com/v2/translate"
            if api_key and api_key.startswith("free:")
            else "https://api.deepl.com/v2/translate"
        )

    async def translate(self, text: str, target_lang: str) -> str:
        async with self.httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                self.endpoint,
                data={
                    "auth_key": self.api_key.replace("free:", ""),
                    "text": text,
                    "target_lang": target_lang.upper(),
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["translations"][0]["text"]

class GoogleAdapter(TranslatorAdapter):
    def __init__(self, project_id: str, location: str = "global"):
        try:
            from google.cloud import translate  # type: ignore
        except ImportError:
            raise RuntimeError("Install google-cloud-translate: pip install google-cloud-translate")
        self.translate = __import__("google.cloud.translate").translate
        self.client = self.translate.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/{location}"

    async def translate(self, text: str, target_lang: str) -> str:
        # google client isn't async; run in thread
        def _call():
            resp = self.client.translate_text(
                contents=[text], target_language_code=target_lang, parent=self.parent
            )
            return resp.translations[0].translated_text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call)

class OfflineAdapter(TranslatorAdapter):
    """Demo offline adapter using HuggingFace Transformers MarianMT for a few pairs."""
    _cache: Dict[str, object] = {}

    def __init__(self):
        try:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError("Install transformers and torch for offline adapter.")
        self.MarianMTModel = __import__("transformers").MarianMTModel
        self.MarianTokenizer = __import__("transformers").MarianTokenizer
        self.torch = __import__("torch")

    def _get_pipeline(self, src: str, tgt: str):
        key = f"{src}->{tgt}"
        if key in self._cache:
            return self._cache[key]
        pair2model = {
            "es->en": "Helsinki-NLP/opus-mt-es-en",
            "en->es": "Helsinki-NLP/opus-mt-en-es",
            "fr->en": "Helsinki-NLP/opus-mt-fr-en",
            "en->fr": "Helsinki-NLP/opus-mt-en-fr",
        }
        model_name = pair2model.get(key)
        if not model_name:
            raise RuntimeError(f"Offline adapter doesn't support pair {key}. Use cloud adapter.")
        tok = self.MarianTokenizer.from_pretrained(model_name)
        model = self.MarianMTModel.from_pretrained(model_name)
        self._cache[key] = (tok, model)
        return tok, model

    async def translate(self, text: str, target_lang: str) -> str:
        src = "en" if target_lang != "en" else "es"  # crude; replace with detector if needed
        tok, model = self._get_pipeline(src, target_lang)
        loop = asyncio.get_running_loop()

        def _run():
            batch = tok([text], return_tensors="pt", padding=True)
            gen = model.generate(**batch, max_new_tokens=400)
            return tok.decode(gen[0], skip_special_tokens=True)

        return await loop.run_in_executor(None, _run)

# -------------- Database --------------
CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS bridges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    src_channel_id INTEGER NOT NULL,
    dst_channel_id INTEGER NOT NULL,
    target_lang TEXT NOT NULL,
    UNIQUE (guild_id, src_channel_id, dst_channel_id)
);
CREATE TABLE IF NOT EXISTS autotranslate_channels (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    target_lang TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id)
);
CREATE TABLE IF NOT EXISTS premium_guilds (
    guild_id INTEGER PRIMARY KEY
);
"""

# -------------- Data Access --------------
class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.executescript(CREATE_TABLES_SQL)
        await self._conn.commit()

    async def add_bridge(self, guild_id: int, src: int, dst: int, lang: str):
        assert self._conn
        await self._conn.execute(
            "INSERT OR IGNORE INTO bridges (guild_id, src_channel_id, dst_channel_id, target_lang) VALUES (?,?,?,?)",
            (guild_id, src, dst, lang),
        )
        await self._conn.commit()

    async def remove_bridge(self, guild_id: int, src: int, dst: int):
        assert self._conn
        await self._conn.execute(
            "DELETE FROM bridges WHERE guild_id=? AND src_channel_id=? AND dst_channel_id=?",
            (guild_id, src, dst),
        )
        await self._conn.commit()

    async def list_bridges(self, guild_id: int) -> List[Tuple[int, int, str]]:
        assert self._conn
        cur = await self._conn.execute(
            "SELECT src_channel_id, dst_channel_id, target_lang FROM bridges WHERE guild_id=?",
            (guild_id,),
        )
        rows = await cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    async def bridge_count(self, guild_id: int) -> int:
        assert self._conn
        cur = await self._conn.execute(
            "SELECT COUNT(*) FROM bridges WHERE guild_id=?",
            (guild_id,),
        )
        (n,) = await cur.fetchone()
        return int(n)

    async def add_autotranslate(self, guild_id: int, channel_id: int, lang: str):
        assert self._conn
        await self._conn.execute(
            "INSERT OR REPLACE INTO autotranslate_channels (guild_id, channel_id, target_lang) VALUES (?,?,?)",
            (guild_id, channel_id, lang),
        )
        await self._conn.commit()

    async def remove_autotranslate(self, guild_id: int, channel_id: int):
        assert self._conn
        await self._conn.execute(
            "DELETE FROM autotranslate_channels WHERE guild_id=? AND channel_id=?",
            (guild_id, channel_id),
        )
        await self._conn.commit()

    async def get_autotranslate(self, guild_id: int) -> Dict[int, str]:
        assert self._conn
        cur = await self._conn.execute(
            "SELECT channel_id, target_lang FROM autotranslate_channels WHERE guild_id=?",
            (guild_id,),
        )
        rows = await cur.fetchall()
        return {int(r[0]): r[1] for r in rows}

    async def grant_premium(self, guild_id: int):
        assert self._conn
        await self._conn.execute(
            "INSERT OR IGNORE INTO premium_guilds (guild_id) VALUES (?)", (guild_id,)
        )
        await self._conn.commit()

    async def revoke_premium(self, guild_id: int):
        assert self._conn
        await self._conn.execute(
            "DELETE FROM premium_guilds WHERE guild_id=?", (guild_id,)
        )
        await self._conn.commit()

    async def is_premium(self, guild_id: int) -> bool:
        assert self._conn
        cur = await self._conn.execute(
            "SELECT 1 FROM premium_guilds WHERE guild_id=?", (guild_id,)
        )
        row = await cur.fetchone()
        return row is not None

# -------------- Bot --------------
INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.messages = True
INTENTS.guilds = True
INTENTS.members = False

class TranslatorCog(commands.Cog):
    def __init__(self, bot: commands.Bot, store: Store, adapter: TranslatorAdapter):
        self.bot = bot
        self.store = store
        self.adapter = adapter
        self.autotranslate_cache: Dict[int, Dict[int, str]] = {}  # guild_id -> {channel_id: lang}
        self.BOT_TAG = "[mirror]"  # not used in webhook mode, but kept for compatibility
        # NEW: cache a webhook per destination channel
        self._webhook_cache: Dict[int, discord.Webhook] = {}

    async def refresh_autotranslate(self, guild_id: int):
        self.autotranslate_cache[guild_id] = await self.store.get_autotranslate(guild_id)

    # ---------- Utility ----------
    async def _is_premium(self, guild: discord.Guild) -> bool:
        if OWNER_GUILD_ID and guild.id == OWNER_GUILD_ID:
            return True
        return await self.store.is_premium(guild.id)

    def _sanitize(self, content: str) -> str:
        return content.strip()

    async def _get_or_create_webhook(self, channel: discord.TextChannel) -> discord.Webhook:
        """Return a reusable webhook for a channel (creates one if missing)."""
        wh = self._webhook_cache.get(channel.id)
        if wh:
            try:
                await wh.fetch()  # ensure it's still valid
                return wh
            except Exception:
                self._webhook_cache.pop(channel.id, None)

        # Try to reuse an existing one we created earlier
        try:
            for existing in await channel.webhooks():
                if existing.name == "MirrorBridge":
                    self._webhook_cache[channel.id] = existing
                    return existing
        except Exception:
            pass

        # Create a new one (requires Manage Webhooks)
        new_wh = await channel.create_webhook(name="MirrorBridge")
        self._webhook_cache[channel.id] = new_wh
        return new_wh

    # ---------- Events ----------
    @commands.Cog.listener()
    async def on_ready(self):
        log.info(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        # Preload autotranslate cache
        for guild in self.bot.guilds:
            await self.refresh_autotranslate(guild.id)
        try:
            synced = await self.bot.tree.sync()
            log.info(f"Slash commands synced: {len(synced)}")
        except Exception as e:
            log.exception("Slash sync failed: %s", e)

    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild):
        await self.refresh_autotranslate(guild.id)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Prevent loops & ignore bots/webhook posts
        if message.author.bot or (message.webhook_id is not None):
            return

        guild = message.guild
        if not guild:
            return

        # Mirroring based on explicit bridges
        bridges = await self.store.list_bridges(guild.id)
        channel_bridges = [(src, dst, lang) for (src, dst, lang) in bridges if src == message.channel.id]
        for _src, dst, lang in channel_bridges:
            await self._mirror(message, dst, lang)

        # Autotranslate destination within same channel
        gmap = self.autotranslate_cache.get(guild.id, {})
        if message.channel.id in gmap:
            lang = gmap[message.channel.id]
            await self._translate_inline(message, lang)

    async def _mirror(self, message: discord.Message, dst_channel_id: int, target_lang: str):
        dst = self.bot.get_channel(dst_channel_id)
        if not isinstance(dst, discord.TextChannel):
            return
        try:
            translated = await self.adapter.translate(self._sanitize(message.content), target_lang)
        except Exception as e:
            log.warning("Translate failed: %s", e)
            return

        # Send THROUGH WEBHOOK as the original author (name + avatar)
        try:
            wh = await self._get_or_create_webhook(dst)

            author_name = message.author.display_name
            author_avatar = (
                message.author.display_avatar.url
                if getattr(message.author, "display_avatar", None)
                else None
            )

            files = []
            if message.attachments:
                for a in message.attachments:
                    try:
                        blob = await a.read()
                        files.append(discord.File(io.BytesIO(blob), filename=a.filename))
                    except Exception:
                        pass

            await wh.send(
                content=(translated or None),
                username=author_name,
                avatar_url=author_avatar,
                files=(files or None),
                allowed_mentions=AllowedMentions.none(),
            )
        except discord.Forbidden:
            # Fallback to normal send if we lack Manage Webhooks
            try:
                await dst.send(
                    translated,
                    allowed_mentions=AllowedMentions.none(),
                )
            except Exception as e:
                log.warning("Mirror send fallback failed: %s", e)
        except Exception as e:
            log.warning("Webhook send failed: %s", e)

    async def _translate_inline(self, message: discord.Message, target_lang: str):
        try:
            translated = await self.adapter.translate(self._sanitize(message.content), target_lang)
            await message.reply(f"`{target_lang}`\n{translated}", mention_author=False)
        except Exception as e:
            log.warning("Inline translate failed: %s", e)

    # ---------- Slash Commands ----------
    @app_commands.command(name="bridge_add", description="Bridge this channel to another channel with target language")
    @app_commands.describe(destination_channel="Target channel to mirror into", target_language="ISO code, e.g., en, es, fr, ja")
    async def bridge_add(self, interaction: discord.Interaction, destination_channel: discord.TextChannel, target_language: str):
        assert interaction.guild
        guild = interaction.guild
        # Premium gating
        premium = await self._is_premium(guild)
        if not premium:
            count = await self.store.bridge_count(guild.id)
            if count >= MAX_BRIDGES_FREE:
                await interaction.response.send_message(
                    f"This server reached the free limit of {MAX_BRIDGES_FREE} bridges. Ask the owner to grant premium.",
                    ephemeral=True,
                )
                return
        await self.store.add_bridge(guild.id, interaction.channel_id, destination_channel.id, target_language)
        await interaction.response.send_message(
            f"Bridge added: #{interaction.channel.name} → #{destination_channel.name} ({target_language})"
        )

    @app_commands.command(name="bridge_remove", description="Remove a bridge from this channel to a destination channel")
    async def bridge_remove(self, interaction: discord.Interaction, destination_channel: discord.TextChannel):
        assert interaction.guild
        await self.store.remove_bridge(interaction.guild.id, interaction.channel_id, destination_channel.id)
        await interaction.response.send_message(
            f"Bridge removed: #{interaction.channel.name} → #{destination_channel.name}"
        )

    @app_commands.command(name="bridge_list", description="List all bridges in this server")
    async def bridge_list(self, interaction: discord.Interaction):
        assert interaction.guild
        bridges = await self.store.list_bridges(interaction.guild.id)
        if not bridges:
            await interaction.response.send_message("No bridges configured yet.")
            return
        lines = []
        for src, dst, lang in bridges:
            src_ch = interaction.guild.get_channel(src)
            dst_ch = interaction.guild.get_channel(dst)
            if isinstance(src_ch, discord.TextChannel) and isinstance(dst_ch, discord.TextChannel):
                lines.append(f"#{src_ch.name} → #{dst_ch.name} ({lang})")
        await interaction.response.send_message("Bridges:\n" + "\n".join(lines))

    @app_commands.command(name="autotranslate", description="Enable autotranslate replies in this channel to a target language")
    async def autotranslate(self, interaction: discord.Interaction, target_language: str):
        assert interaction.guild
        guild = interaction.guild
        premium = await self._is_premium(guild)
        if not premium:
            await interaction.response.send_message("Autotranslate requires premium on this server.", ephemeral=True)
            return
        await self.store.add_autotranslate(guild.id, interaction.channel_id, target_language)
        await self.refresh_autotranslate(guild.id)
        await interaction.response.send_message(f"Autotranslate enabled here → {target_language}.")

    @app_commands.command(name="autotranslate_off", description="Disable autotranslate in this channel")
    async def autotranslate_off(self, interaction: discord.Interaction):
        assert interaction.guild
        await self.store.remove_autotranslate(interaction.guild.id, interaction.channel_id)
        await self.refresh_autotranslate(interaction.guild.id)
        await interaction.response.send_message("Autotranslate disabled for this channel.")

    # Premium admin (owner only)
    @app_commands.command(name="premium_grant", description="Grant premium to this server (owner only)")
    async def premium_grant(self, interaction: discord.Interaction):
        if interaction.user.id != OWNER_USER_ID:
            await interaction.response.send_message("Only the bot owner can grant premium.", ephemeral=True)
            return
        assert interaction.guild
        await self.store.grant_premium(interaction.guild.id)
        await interaction.response.send_message("Premium granted for this server.")

    @app_commands.command(name="premium_revoke", description="Revoke premium from this server (owner only)")
    async def premium_revoke(self, interaction: discord.Interaction):
        if interaction.user.id != OWNER_USER_ID:
            await interaction.response.send_message("Only the bot owner can revoke premium.", ephemeral=True)
            return
        assert interaction.guild
        await self.store.revoke_premium(interaction.guild.id)
        await interaction.response.send_message("Premium revoked for this server.")

    @app_commands.command(name="premium_status", description="Show whether this server has premium")
    async def premium_status(self, interaction: discord.Interaction):
        assert interaction.guild
        is_p = await self._is_premium(interaction.guild)
        await interaction.response.send_message(f"Premium: {'Yes' if is_p else 'No'}")

    @app_commands.command(name="lang_list", description="List common ISO language codes")
    async def lang_list(self, interaction: discord.Interaction):
        codes = "en, es, fr, de, it, pt, ja, ko, zh, ru, ar, hi"
        await interaction.response.send_message(f"Common target language codes: {codes}")

# -------------- Bootstrap --------------
async def get_adapter() -> TranslatorAdapter:
    if TRANSLATOR_BACKEND == "deepl":
        if not DEEPL_API_KEY:
            raise RuntimeError("DEEPL_API_KEY not set")
        return DeepLAdapter(DEEPL_API_KEY)
    elif TRANSLATOR_BACKEND == "google":
        if not GOOGLE_PROJECT_ID:
            raise RuntimeError("GOOGLE_PROJECT_ID not set")
        return GoogleAdapter(GOOGLE_PROJECT_ID, GOOGLE_LOCATION)
    elif TRANSLATOR_BACKEND == "offline":
        return OfflineAdapter()
    else:
        raise RuntimeError(f"Unknown TRANSLATOR_BACKEND: {TRANSLATOR_BACKEND}")

async def main():
    if not BOT_TOKEN:
        raise SystemExit("DISCORD_TOKEN not set")
    store = Store(DB_PATH)
    await store.init()
    adapter = await get_adapter()

    bot = commands.Bot(command_prefix="!", intents=INTENTS)
    cog = TranslatorCog(bot, store, adapter)
    await bot.add_cog(cog)
    await bot.start(BOT_TOKEN)

if __name__ == "__main__":
    """
    Quick start:
    1) pip install -U discord.py aiosqlite httpx
       # For DeepL: set DEEPL_API_KEY (use prefix free: for free keys)
       # Or: pip install google-cloud-translate and set GOOGLE_* envs
    2) export DISCORD_TOKEN=your_bot_token
       export OWNER_USER_ID=your_discord_user_id
       export OWNER_GUILD_ID=your_server_id  # optional
       export TRANSLATOR_BACKEND=deepl  # or google|offline
       export DEEPL_API_KEY=free:xxxxx  # if using DeepL free
    3) python main.py

    Permissions needed in destination channels:
      View Channels, Read Message History, Attach Files, Embed Links,
      Use Slash Commands, **Manage Webhooks** (for author-style mirroring)

    Usage examples:
      /bridge_add destination_channel:#spanish target_language:es
      /bridge_list
      /autotranslate target_language:en
      /autotranslate_off
      /premium_grant (owner only)
    """
    asyncio.run(main())
