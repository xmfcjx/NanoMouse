"""Read-mostly simulated equipment gateway used by the EdgeOps prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EquipmentGateway:
    def __init__(self, data_dir: str) -> None:
        root = Path(data_dir)
        self.devices = self._load_json(root / "devices.json", {})
        self.error_codes = self._load_json(root / "error_codes.json", {})
        self.maintenance = self._load_json(root / "maintenance.json", {})
        self.work_order_drafts: List[Dict[str, Any]] = []

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        value = self.devices.get(device_id.upper())
        return dict(value) if value else None

    def lookup_error_code(self, error_code: str) -> Optional[Dict[str, Any]]:
        value = self.error_codes.get(error_code.upper())
        return dict(value) if value else None

    def get_maintenance_history(self, device_id: str) -> List[Dict[str, Any]]:
        return list(self.maintenance.get(device_id.upper(), []))

    def locate_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        value = self.devices.get(asset_id.upper())
        if not value:
            return None
        return {
            "asset_id": asset_id.upper(),
            "location": value.get("location"),
            "last_seen": value.get("last_seen"),
            "status": value.get("status"),
        }

    def create_work_order_draft(
        self, device_id: str, title: str, description: str
    ) -> Dict[str, Any]:
        draft = {
            "draft_id": "WO-DRAFT-%04d" % (len(self.work_order_drafts) + 1),
            "device_id": device_id.upper(),
            "title": title,
            "description": description,
            "status": "draft",
        }
        self.work_order_drafts.append(draft)
        return dict(draft)
