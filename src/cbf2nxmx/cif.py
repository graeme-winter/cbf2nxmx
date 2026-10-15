"""A small CIF 1.1 reader for the text part of a CBF.

It handles data blocks, single items, loops, quoted strings, semicolon text
fields and comments. That is all the imgCIF header of a CBF uses. Values are
kept as strings; '.' and '?' become None.

The result is a mapping from category (lower case, without the leading
underscore) to a list of rows, each row a dict from item name (lower case) to
value. A single item `_cat.item value` becomes a one-row table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CifBlock", "parse", "tokenise"]


@dataclass
class CifBlock:
    name: str
    tables: dict[str, list[dict[str, str | None]]] = field(default_factory=dict)

    def table(self, category: str) -> list[dict[str, str | None]]:
        return self.tables.get(category.lower().lstrip("_"), [])

    def has(self, category: str) -> bool:
        return category.lower().lstrip("_") in self.tables

    def value(self, category: str, item: str, row: int = 0) -> str | None:
        rows = self.table(category)
        if len(rows) <= row:
            return None
        return rows[row].get(item.lower())

    def single(self, category: str, item: str) -> str | None:
        """The item's value, which must be the same in every row."""
        rows = self.table(category)
        values = {r.get(item.lower()) for r in rows}
        if len(values) > 1:
            raise ValueError(
                f"_{category}.{item} has several values {sorted(map(str, values))}"
            )
        return values.pop() if values else None


# A token is (kind, text): kind is "tag", "value" or "keyword".
def tokenise(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            # A text field: to the next line beginning with ';'.
            body = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                body.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError("unterminated semicolon text field")
            text_value = "\n".join(body)
            if text_value.startswith("\n"):
                text_value = text_value[1:]
            tokens.append(("value", text_value))
            i += 1
            continue
        j = 0
        n = len(line)
        while j < n:
            c = line[j]
            if c.isspace():
                j += 1
                continue
            if c == "#":
                break
            if c in "'\"":
                # Closed by the same quote followed by whitespace or the end.
                k = j + 1
                while k < n:
                    if line[k] == c and (k + 1 == n or line[k + 1].isspace()):
                        break
                    k += 1
                if k >= n:
                    raise ValueError(f"unterminated quoted string: {line!r}")
                tokens.append(("value", line[j + 1 : k]))
                j = k + 1
                continue
            k = j
            while k < n and not line[k].isspace():
                k += 1
            word = line[j:k]
            low = word.lower()
            if word.startswith("_"):
                tokens.append(("tag", word))
            elif low.startswith(("data_", "save_", "global_", "stop_")) or low == "loop_":
                tokens.append(("keyword", word))
            else:
                tokens.append(("value", word))
            j = k
        i += 1
    return tokens


def _split_tag(tag: str) -> tuple[str, str]:
    t = tag.lower().lstrip("_")
    if "." not in t:
        return t, ""
    cat, item = t.split(".", 1)
    return cat, item


def _norm(value: str, quoted_flag: bool = False) -> str | None:
    if value in (".", "?"):
        return None
    return value


def parse(text: str) -> list[CifBlock]:
    """Parse CIF text into its data blocks."""
    tokens = tokenise(text)
    blocks: list[CifBlock] = []
    block: CifBlock | None = None
    i = 0
    nt = len(tokens)

    def current() -> CifBlock:
        nonlocal block
        if block is None:
            block = CifBlock(name="")
            blocks.append(block)
        return block

    while i < nt:
        kind, text_ = tokens[i]
        if kind == "keyword" and text_.lower().startswith("data_"):
            block = CifBlock(name=text_[5:])
            blocks.append(block)
            i += 1
        elif kind == "keyword" and text_.lower() == "loop_":
            i += 1
            tags = []
            while i < nt and tokens[i][0] == "tag":
                tags.append(tokens[i][1])
                i += 1
            values = []
            while i < nt and tokens[i][0] == "value":
                values.append(tokens[i][1])
                i += 1
            if not tags:
                raise ValueError("loop_ without tags")
            if len(values) % len(tags):
                raise ValueError(
                    f"loop of {tags[0]}...: {len(values)} values for {len(tags)} tags"
                )
            cats = {_split_tag(t)[0] for t in tags}
            if len(cats) != 1:
                raise ValueError(f"loop mixes categories {sorted(cats)}")
            cat = cats.pop()
            items = [_split_tag(t)[1] for t in tags]
            rows = current().tables.setdefault(cat, [])
            for r in range(0, len(values), len(tags)):
                rows.append(
                    {it: _norm(v) for it, v in zip(items, values[r : r + len(tags)])}
                )
        elif kind == "tag":
            if i + 1 >= nt or tokens[i + 1][0] != "value":
                raise ValueError(f"no value for {text_}")
            cat, item = _split_tag(text_)
            rows = current().tables.setdefault(cat, [])
            if not rows:
                rows.append({})
            rows[0][item] = _norm(tokens[i + 1][1])
            i += 2
        else:
            # A stray value or save_ frame keyword: skipped.
            i += 1
    return blocks
