"""constants.py — Class label mappings for each parasite dataset."""

CLASS_NAMES: dict[str, list[str]] = {
    "eggs": [
        "Hymenolepis nana",
        "Hymenolepis diminuta",
        "Ancylostoma",
        "Enterobius vermicularis",
        "Ascaris lumbricoides",
        "Trichuris trichiura",
        "Schistosoma mansoni",
        "Taenia spp",
        "Impurities",
    ],
    "larvae": [
        "Strongyloides stercoralis",
        "Impurities",
    ],
    "protozoan": [
        "Entamoeba coli",
        "Entamoeba histolytica",
        "Endolimax nana",
        "Giardia intestinalis",
        "Iodamoeba bütschlii",
        "Blastocystis hominis",
        "Impurities",
    ],
}
