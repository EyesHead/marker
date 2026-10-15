"""Durable intermediate documents and validated recognition jobs; no credentials."""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict

from marker.cloud.content import ContentResult
from marker.cloud.context import RecognitionFailure
from marker.cloud.resources import retain_png


PIPELINE_VERSION = "whole-object-v1"


class CachedResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["text", "math", "figure", "table", "image", "unknown"]
    markdown: str
    html: str
    complete: Literal[True]
    reasons: list[str]
    latex: str | None
    source_atom_ids: list[str]
    assets: dict[str, str]


def _canonical(record):
    return json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _validate_cached_content(record):
    """Cache is input, not a substitute for the output guards."""
    from bs4 import BeautifulSoup
    from marker.cloud.content import latex_to_mathml

    for field in ("html", "markdown"):
        soup = BeautifulSoup(record[field], "html.parser")
        if soup.find(["script", "style", "iframe", "object", "embed", "link", "svg"]):
            raise ValueError("unsafe cached markup")
        for node in soup.find_all(True):
            if any(str(key).lower().startswith("on") for key in node.attrs):
                raise ValueError("unsafe cached attribute")
        for image in soup.find_all("img"):
            if image.get("src") not in record["assets"]:
                raise ValueError("unknown cached image")
        formulas = list(soup.find_all("math"))
        for math in formulas:
            annotation = math.find(
                "annotation", attrs={"encoding": "application/x-tex"}
            )
            if annotation is None:
                raise ValueError("cached math has no original LaTeX")
            math.replace_with(
                BeautifulSoup(
                    latex_to_mathml(
                        annotation.get_text(), display=math.get("display", "inline")
                    ),
                    "html.parser",
                )
            )
        if formulas:
            record[field] = str(soup)
    if record["kind"] == "math":
        latex = record["latex"]
        latex_to_mathml(latex)
        if record["markdown"] not in ("$" + latex + "$", "$$" + latex + "$$"):
            raise ValueError("cached math lost its exact LaTeX")
    if record["kind"] == "table":
        from marker.cloud.whole_tables import normalize_table_html

        soup = BeautifulSoup(record["html"], "html.parser")
        mapping = {}
        for index, image in enumerate(soup.find_all("img")):
            identity = f"cached{index}"
            item = {"name": image["src"]}
            style = image.get("style", "")
            width = re.search(r"(?:^|;)\s*width\s*:\s*(\d+(?:\.\d+)?)pt", style)
            height = re.search(r"(?:^|;)\s*height\s*:\s*(\d+(?:\.\d+)?)pt", style)
            if width and height:
                item["bbox"] = (0.0, 0.0, float(width[1]), float(height[1]))
            mapping[identity] = item
            image.attrs = {"data-source-image": identity, "alt": "Figure"}
        if record["markdown"] != record["html"]:
            raise ValueError("cached table markup differs")
        normalized = normalize_table_html(str(soup), source_images=mapping)
        record["html"] = record["markdown"] = normalized
    if record["reasons"] or not record["markdown"].strip():
        raise ValueError("empty cached recognition")


class StageStore:
    def __init__(self, root, pdf, signature):
        digest = hashlib.sha256()
        with open(pdf, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.input_hash = digest.hexdigest()
        digest.update(
            json.dumps(
                {"version": PIPELINE_VERSION, "config": signature},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        )
        self.root = Path(root) / digest.hexdigest()[:32]
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, name, text):
        if Path(name).name != name:
            raise RecognitionFailure("Intermediate filename is invalid")
        self._write(self.root / name, text.encode("utf-8"))

    @staticmethod
    def _write(path, data):
        for field in ("OLLAMA_API_KEY", "OPENAI_API_KEY"):
            key = os.environ.get(field)
            if key and key.encode() in data:
                raise RecognitionFailure(
                    "Recognition contained a credential; result was not stored"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    def _job(self, identity):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", identity):
            raise RecognitionFailure("Intermediate job identity is invalid")
        return self.root / "jobs" / identity

    def save_source(self, identity, image, index=0):
        path = self._job(identity) / f"source-{index}.png"
        stream = BytesIO()
        image.save(stream, "PNG")
        self._write(path, stream.getvalue())
        return path.relative_to(self.root).as_posix()

    def save(self, identity, result):
        if not result.complete or result.reasons:
            return
        folder = self._job(identity)
        assets = {}
        for name, image in result.resources.items():
            if Path(name).name != name or not name.lower().endswith(".png"):
                raise RecognitionFailure("Recognition resource filename is invalid")
            stream = BytesIO()
            image.save(stream, "PNG")
            data = stream.getvalue()
            self._write(folder / name, data)
            assets[name] = hashlib.sha256(data).hexdigest()
        record = {
            field: getattr(result, field)
            for field in (
                "kind",
                "markdown",
                "html",
                "complete",
                "reasons",
                "latex",
                "source_atom_ids",
            )
        }
        record["assets"] = assets
        # This checksum detects accidental corruption; it is not an
        # authentication mechanism. Content is validated again when loaded.
        envelope = {
            "record": record,
            "job_id": identity,
            "scope": self.root.name,
            "sha256": hashlib.sha256(_canonical(record)).hexdigest(),
        }
        self._write(
            folder / "result.json", json.dumps(envelope, ensure_ascii=False).encode()
        )

    def load(self, identity):
        folder = self._job(identity)
        try:
            envelope = json.loads((folder / "result.json").read_text())
            if not isinstance(envelope, dict) or not isinstance(
                envelope.get("record"), dict
            ):
                return None
            if (
                envelope.get("job_id") != identity
                or envelope.get("scope") != self.root.name
            ):
                return None
            record = envelope["record"]
            if hashlib.sha256(_canonical(record)).hexdigest() != envelope.get("sha256"):
                return None
            record = CachedResult.model_validate(record).model_dump()
            _validate_cached_content(record)
            assets = {}
            for name, expected in record.pop("assets").items():
                if Path(name).name != name or not name.lower().endswith(".png"):
                    return None
                data = (folder / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    return None
                with Image.open(BytesIO(data)) as image:
                    assets[name] = retain_png(image)
            record["resources"] = assets
            record["reasons"] = tuple(record["reasons"])
            record["source_atom_ids"] = tuple(record["source_atom_ids"])
            return ContentResult(**record)
        except (OSError, ValueError, TypeError, KeyError, RecognitionFailure):
            return None
