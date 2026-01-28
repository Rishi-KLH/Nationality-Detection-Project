def map_nationality(race, confidence, threshold=0.60):
    # ✅ If model confidence is low, avoid wrong mapping
    if confidence < threshold:
        return "Other"

    race = race.lower()

    if "indian" in race:
        return "Indian"
    elif "black" in race:
        return "African"
    elif "white" in race:
        return "United States"
    else:
        return "Other"