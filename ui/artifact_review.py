"""Async rendezvous for one provisional Plan or Goal review."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class PendingArtifactReview:
    """Tie one visible provisional artifact to its suspended finalizer."""

    kind: Literal["plan", "goal"]
    artifact: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]

    @classmethod
    def create(
        cls,
        kind: Literal["plan", "goal"],
        artifact: dict[str, Any],
    ) -> "PendingArtifactReview":
        return cls(kind, artifact, asyncio.get_running_loop().create_future())

    def matches(self, kind: str, artifact_id: str) -> bool:
        return self.kind == kind and str(self.artifact.get("id") or "") == artifact_id

    def resolve(self, action: str, **values: Any) -> None:
        if not self.future.done():
            self.future.set_result({"action": action, "artifact": self.artifact, **values})

    def cancel(self) -> None:
        if not self.future.done():
            self.future.cancel()

    async def wait(self) -> dict[str, Any]:
        return await self.future
