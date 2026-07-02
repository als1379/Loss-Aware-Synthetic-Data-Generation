from lossaware.datasets import get_loader, list_datasets


def test_adult_loader_is_registered():
    loader = get_loader("adult")

    assert loader.name == "adult"


def test_candidate_datasets_are_listed():
    datasets = list_datasets()

    assert "uci_bank_marketing" in datasets
    assert datasets["uci_bank_marketing"]["implemented"] is False


def test_folktables_loader_is_registered():
    loader = get_loader("folktables_acs_income")

    assert loader.name == "folktables_acs_income"
