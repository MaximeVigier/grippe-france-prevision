"""Tests des utilitaires d'acquisition et de calendrier."""
from src.data_loader import yearweek_to_date


def test_yearweek_to_date_lundi_iso():
    d = yearweek_to_date(202634)               # semaine ISO 34 de 2026
    assert (d.year, d.month, d.day) == (2026, 8, 17)
    assert d.weekday() == 0                    # lundi


def test_yearweek_accepte_str_et_int():
    assert yearweek_to_date("201901") == yearweek_to_date(201901)


def test_semaine_53():
    # 2020 est une année ISO à 53 semaines
    d = yearweek_to_date(202053)
    assert d.year == 2020 and d.month == 12
