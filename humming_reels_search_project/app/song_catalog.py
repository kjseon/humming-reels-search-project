# song_catalog.py - local filename stems mapped to real song metadata

SONG_CATALOG = {
    "beauty": {
        "catalog_key": "beauty",
        "original_file_stem": "Beauty_original",
        "display_title": "Beauty And A Beat",
        "artist": "Justin Bieber",
        "featured_artist": "Nicki Minaj",
        "full_title": "Beauty And A Beat (feat. Nicki Minaj)",
        "youtube_query": "Justin Bieber Beauty And A Beat ft Nicki Minaj",
    },
    "rude": {
        "catalog_key": "rude",
        "original_file_stem": "rude_original",
        "display_title": "RUDE!",
        "artist": "Hearts2Hearts",
        "featured_artist": "",
        "full_title": "RUDE!",
        "youtube_query": "Hearts2Hearts RUDE!",
    }
}


def get_song_metadata(song_name: str) -> dict:
    """Return real-world metadata for a parsed local song/version name."""
    normalized = song_name.replace("_", " ").lower()
    for key, metadata in SONG_CATALOG.items():
        if key in normalized:
            return metadata

    return {
        "catalog_key": normalized.replace(" ", "_"),
        "original_file_stem": song_name.replace(" ", "_"),
        "display_title": song_name,
        "artist": "",
        "featured_artist": "",
        "full_title": song_name,
        "youtube_query": song_name,
    }
