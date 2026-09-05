from __future__ import annotations

import random

from astroswarm_rl.league import League, LeagueEntry


def test_league_deduplicates_samples_and_updates_ratings(tmp_path) -> None:
    league = League(tmp_path)
    league.add(LeagueEntry("attacker", 10, "checkpoints/10.pt"))
    league.add(LeagueEntry("attacker", 10, "checkpoints/10.pt", 1010.0))

    assert len(league.entries) == 1
    entry = league.sample("attacker", random.Random(1))
    assert entry is not None
    old_rating = league.update_rating(entry, score=1.0, opponent_rating=1000.0)

    assert old_rating == 1010.0
    assert entry.rating > old_rating
    loaded = League(tmp_path)
    assert loaded.entries[0].rating == entry.rating
    assert loaded.summary()["roles"]["attacker"]["snapshots"] == 1
