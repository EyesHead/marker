"""Durable, pessimistic USD reservations for one PDF across OpenAI models.

The caller chooses a stable ledger path keyed by the PDF's SHA256. Reservations
are written before transport and remain charged without provider usage, even
after a timeout or a process restart. ``path=None`` is explicitly in-memory.
Only token counts, model names and monetary amounts are stored; never prompts,
document contents, credentials, or responses. Prices are per million tokens.
"""

from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock, RLock
from uuid import uuid4


MAX_DOCUMENT_USD = Decimal("1.00")
DEFAULT_LIMIT_USD = "1.00"
DEFAULT_MODEL = "gpt-6-luna"
LONG_CONTEXT_THRESHOLD = 272000
PRICING_VERSION = "openai-2026-10-16"
_MILLION = Decimal(1000000)
_ZERO = Decimal(0)
_PRICES = {
    "gpt-6-luna": tuple(map(Decimal, ("0.10", "0.50", "0.01", "0.125"))),
    "gpt-6.1-sol": tuple(map(Decimal, ("2", "10", "0.10", "2.50"))),
}
_PATH_LOCKS = {}
_PATH_LOCKS_GUARD = Lock()


class BudgetError(RuntimeError):
    """A safe budget failure that must prevent an unreserved provider call."""


class BudgetExceeded(BudgetError):
    """A document has insufficient uncommitted money for this request."""


class BudgetCorruptionError(BudgetError):
    """The durable ledger cannot safely establish previously committed cost."""


def _money(value):
    if not isinstance(value, (str, Decimal)):
        raise BudgetError("USD amounts must be decimal strings or Decimal values")
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError):
        raise BudgetError("USD amount is invalid") from None
    if not amount.is_finite() or amount < 0:
        raise BudgetError("USD amount must be finite and nonnegative")
    return amount


def _limit(value):
    amount = _money(value)
    if amount <= 0 or amount > MAX_DOCUMENT_USD:
        raise BudgetError("Document budget must be above zero and cannot exceed USD 1.00")
    return amount


def _tokens(value):
    if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
        raise BudgetError("Token counts must be nonnegative integers")
    return value


def _cost(model, input_tokens, output_tokens, cached_input_tokens=0, cache_write_tokens=0, *, reserve=False):
    if not isinstance(model, str) or model not in _PRICES:
        raise BudgetError("Model has no verified USD price; request is blocked")
    counts = [_tokens(value) for value in
              (input_tokens, output_tokens, cached_input_tokens, cache_write_tokens)]
    incoming, outgoing, cached, written = counts
    if cached + written > incoming:
        raise BudgetError("Cached and cache-write tokens exceed total input tokens")
    normal_rate, output_rate, cached_rate, write_rate = _PRICES[model]
    if incoming > LONG_CONTEXT_THRESHOLD:
        normal_rate *= 2
        cached_rate *= 2
        write_rate *= 2
        output_rate *= Decimal("1.5")
    with localcontext() as context:
        context.prec = 80
        if reserve:
            # An automatic cache write may bill more than ordinary input. Do
            # not assume a cache hit or charge this cheaper on an unknown call.
            incoming_cost = Decimal(incoming) * max(normal_rate, write_rate)
        else:
            incoming_cost = (Decimal(incoming - cached - written) * normal_rate
                             + Decimal(cached) * cached_rate
                             + Decimal(written) * write_rate)
        return (incoming_cost + Decimal(outgoing) * output_rate) / _MILLION


def _amount_text(value):
    if value and value.adjusted() < -18:
        return str(value)
    return format(value, "f")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BudgetCorruptionError("Budget ledger contains duplicate fields")
        result[key] = value
    return result


class DollarBudget:
    """One lifetime document ledger shared by launches, models and processes.

    Input usage includes cached and cache-write subsets. The long-context tier
    applies to the request's total input count, including those subsets. A
    lowered persisted limit cannot be silently raised by another instance.
    ``spent_usd`` means settled provider usage; ``committed_usd`` also includes
    all unresolved reservations and is the value enforced against the limit.
    """

    def __init__(self, path: Path | None, limit_usd: str | Decimal = DEFAULT_LIMIT_USD):
        self.limit_usd = _limit(limit_usd)
        self.path = Path(path).expanduser().resolve() if path is not None else None
        if self.path is None:
            self._thread_lock = RLock()
        else:
            with _PATH_LOCKS_GUARD:
                self._thread_lock = _PATH_LOCKS.setdefault(os.path.normcase(str(self.path)), RLock())
        self._memory = None
        self._initialized = False
        with self._transaction():
            pass
        self._initialized = True

    def _new(self):
        return {"version": 1, "pricing_version": PRICING_VERSION,
                "limit_usd": _amount_text(self.limit_usd), "reservations": {}}

    @contextmanager
    def _file_lock(self):
        if self.path is None:
            yield None
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "r+b", buffering=0) as stream:
                if os.fstat(stream.fileno()).st_size == 0:
                    stream.write(b"0")
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield stream
                finally:
                    stream.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            raise BudgetError("Cannot lock or persist document budget") from None

    def _load(self, lock_stream):
        if self.path is None:
            return deepcopy(self._memory) if self._memory is not None else self._new()
        if not self.path.exists():
            lock_stream.seek(0)
            if self._initialized or lock_stream.read(1) == b"1":
                raise BudgetCorruptionError("Previously initialized budget ledger is missing")
            return self._new()
        try:
            if self.path.stat().st_size > 32 * 1024 * 1024:
                raise BudgetCorruptionError("Budget ledger exceeds its safe size limit")
            envelope = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
            if not isinstance(envelope, dict) or set(envelope) != {"ledger", "sha256"}:
                raise ValueError
            ledger = envelope["ledger"]
            if envelope["sha256"] != hashlib.sha256(_canonical(ledger)).hexdigest():
                raise ValueError
            self._validate(ledger)
            return ledger
        except BudgetCorruptionError:
            raise
        except (OSError, ValueError, TypeError, KeyError, BudgetError, InvalidOperation):
            raise BudgetCorruptionError("Budget ledger is unreadable or corrupted; no counter was reset") from None

    @staticmethod
    def _validate(ledger):
        if (not isinstance(ledger, dict)
                or set(ledger) != {"version", "pricing_version", "limit_usd", "reservations"}
                or type(ledger["version"]) is not int or ledger["version"] != 1
                or ledger["pricing_version"] != PRICING_VERSION
                or not isinstance(ledger["reservations"], dict)):
            raise ValueError
        _limit(ledger["limit_usd"])
        for identity, reservation in ledger["reservations"].items():
            if not re.fullmatch(r"[0-9a-f]{32}", identity) or not isinstance(reservation, dict):
                raise ValueError
            fields = {"model", "input_tokens", "max_output_tokens", "reserved_usd", "status"}
            if reservation.get("status") == "settled":
                fields |= {"usage", "actual_usd"}
            elif reservation.get("status") != "reserved":
                raise ValueError
            if set(reservation) != fields or _tokens(reservation["max_output_tokens"]) < 1:
                raise ValueError
            expected = _cost(reservation["model"], reservation["input_tokens"],
                             reservation["max_output_tokens"], reserve=True)
            if _money(reservation["reserved_usd"]) != expected:
                raise ValueError
            if reservation["status"] == "settled":
                usage = reservation["usage"]
                if not isinstance(usage, dict) or set(usage) != {
                        "input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens"}:
                    raise ValueError
                if _money(reservation["actual_usd"]) != _cost(reservation["model"], **usage):
                    raise ValueError

    def _save(self, ledger):
        if self.path is None:
            self._memory = deepcopy(ledger)
            return
        envelope = {"ledger": ledger, "sha256": hashlib.sha256(_canonical(ledger)).hexdigest()}
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix="." + self.path.name + ".", suffix=".tmp", dir=self.path.parent)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical(envelope))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            if os.name != "nt":
                directory = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        except OSError:
            raise BudgetError("Cannot persist document budget; provider call is blocked") from None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    @contextmanager
    def _transaction(self):
        with self._thread_lock, self._file_lock() as lock_stream:
            ledger = self._load(lock_stream)
            previous = _canonical(ledger)
            ledger["limit_usd"] = _amount_text(min(self.limit_usd, _limit(ledger["limit_usd"])))
            try:
                yield ledger
            finally:
                if self.path is None or not self.path.exists() or _canonical(ledger) != previous:
                    self._save(ledger)
                if lock_stream is not None:
                    lock_stream.seek(0)
                    if lock_stream.read(1) != b"1":
                        lock_stream.seek(0)
                        lock_stream.write(b"1")
                        os.fsync(lock_stream.fileno())

    @staticmethod
    def _snapshot(ledger):
        spent = reserved = _ZERO
        settled_count = 0
        tokens = {key: 0 for key in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens")}
        with localcontext() as context:
            context.prec = 80
            for reservation in ledger["reservations"].values():
                if reservation["status"] == "settled":
                    spent += _money(reservation["actual_usd"])
                    settled_count += 1
                    for key in tokens:
                        tokens[key] += reservation["usage"][key]
                else:
                    reserved += _money(reservation["reserved_usd"])
            committed = spent + reserved
            limit = _limit(ledger["limit_usd"])
            return {"limit_usd": _amount_text(limit), "spent_usd": _amount_text(spent),
                    "reserved_usd": _amount_text(reserved), "committed_usd": _amount_text(committed),
                    "remaining_usd": _amount_text(max(_ZERO, limit - committed)),
                    "reservations": len(ledger["reservations"]),
                    "pending_reservations": len(ledger["reservations"]) - settled_count,
                    "settled_reservations": settled_count, "over_limit": committed > limit,
                    "exhausted": committed >= limit, **tokens}

    def reserve(self, input_tokens: int, max_output_tokens: int, model: str = DEFAULT_MODEL) -> str:
        if _tokens(max_output_tokens) < 1:
            raise BudgetError("A positive maximum output token count is required")
        cost = _cost(model, input_tokens, max_output_tokens, reserve=True)
        with self._transaction() as ledger:
            snapshot = self._snapshot(ledger)
            with localcontext() as context:
                context.prec = 80
                if _money(snapshot["committed_usd"]) + cost > _limit(ledger["limit_usd"]):
                    raise BudgetExceeded("Document USD budget exhausted; provider call is blocked")
            identity = uuid4().hex
            ledger["reservations"][identity] = {
                "model": model, "input_tokens": input_tokens, "max_output_tokens": max_output_tokens,
                "reserved_usd": _amount_text(cost), "status": "reserved",
            }
        return identity

    def settle(self, identity, input_tokens, output_tokens, cached_input_tokens=0, cache_write_tokens=0):
        """Charge known provider usage once, even if content validation failed.

        Missing/unknown usage keeps the original pessimistic reservation.
        Unexpected usage above the reserved amount is recorded truthfully and
        may exceed the budget; no subsequent request can spend that overage.
        """
        with self._transaction() as ledger:
            if not isinstance(identity, str) or identity not in ledger["reservations"]:
                raise BudgetError("Unknown budget reservation")
            reservation = ledger["reservations"][identity]
            if any(value is None for value in
                   (input_tokens, output_tokens, cached_input_tokens, cache_write_tokens)):
                return
            usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                     "cached_input_tokens": cached_input_tokens, "cache_write_tokens": cache_write_tokens}
            cost = _cost(reservation["model"], **usage)
            if reservation["status"] == "settled":
                if reservation["usage"] != usage:
                    raise BudgetError("A settled reservation cannot be charged again with different usage")
                return
            reservation.update(status="settled", usage=usage, actual_usd=_amount_text(cost))

    def snapshot(self):
        with self._transaction() as ledger:
            return self._snapshot(ledger)
