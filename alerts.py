# futu_autotrade/alerts.py
"""Simple alerting helpers for Telegram and Email.
Usage:
  from alerts import send_alert
  send_alert('RiskLock', {'symbol': 'US.AAPL', 'reason': 'max daily loss'}, cfg)
Config (config.yaml):
  alerts:
    enabled: true
    throttle_seconds: 60
    telegram:
      bot_token: ""
      chat_id: ""
    email:
      smtp_host: "smtp.example.com"
      smtp_port: 587
      username: ""
      password: ""
      from_addr: "bot@example.com"
      to_addrs: ["you@example.com"]
"""
from __future__ import annotations
import time, json, smtplib, ssl
from typing import Dict, Any, List

_last_sent = {}

def _throttle(key: str, seconds: int) -> bool:
    now = time.time()
    last = _last_sent.get(key, 0)
    if now - last < seconds:
        return True
    _last_sent[key] = now
    return False

def _send_telegram(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    try:
        import requests
    except Exception as e:
        return False, f"requests not installed: {e}"
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
        if r.status_code != 200:
            return False, f"telegram http {r.status_code}: {r.text[:200]}"
        return True, "ok"
    except Exception as e:
        return False, f"telegram error: {e}"

def _send_email(cfg: Dict[str, Any], subject: str, text: str) -> tuple[bool, str]:
    try:
        host = cfg.get('smtp_host'); port = int(cfg.get('smtp_port', 587))
        user = cfg.get('username'); pwd = cfg.get('password')
        from_addr = cfg.get('from_addr'); to_addrs = cfg.get('to_addrs') or []
        if not (host and user and pwd and from_addr and to_addrs):
            return False, "email config incomplete"
        msg = f"From: {from_addr}\nTo: {', '.join(to_addrs)}\nSubject: {subject}\n\n{text}"
        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=5) as server:
            server.starttls(context=ctx)
            server.login(user, pwd)
            server.sendmail(from_addr, to_addrs, msg.encode('utf-8'))
        return True, "ok"
    except Exception as e:
        return False, f"email error: {e}"

def send_alert(event: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    a = (cfg or {}).get('alerts') or {}
    if not a.get('enabled', False):
        return
    throttle = int(a.get('throttle_seconds', 60))
    key = f"{event}:{payload.get('symbol','*')}"
    if _throttle(key, throttle):
        return
    text = f"[{event}]\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    # Telegram first
    tcfg = a.get('telegram') or {}
    if tcfg.get('bot_token') and tcfg.get('chat_id'):
        _send_telegram(tcfg['bot_token'], tcfg['chat_id'], text)
    # Email fallback
    ecfg = a.get('email') or {}
    if ecfg:
        _send_email(ecfg, f"[Alert] {event}", text)
