from dataclasses import dataclass, field
from typing import Any

@dataclass
class MediaMetadata:
    media_id: str
    media_mime_type: str # MIME type of the media (eg. "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf")
    media_filename: str # Filename of the media (eg. "image.png", "document.pdf")
    media_extension: str # Extension of the media (eg. "png", "pdf")
    media_size: int # Size of the media in bytes
    storage_type: str # Type of the storage (eg. "local", "s3", "cloudflare_r2")
    storage_location: str # Location of the media in the storage (eg. URL or file path)
    
    extras: dict[str, Any] = field(default_factory=dict)    # Backend-specific extension point for storing custom data (e.g. S3 bucket/key). It has to JSON serializable.
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "media_mime_type": self.media_mime_type,
            "media_filename": self.media_filename,
            "media_extension": self.media_extension,
            "media_size": self.media_size,
            "storage_type": self.storage_type,
            "storage_location": self.storage_location,
            "extras": self.extras,
        }
        
