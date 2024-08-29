def find_closest_mins2(list_of_peaks: list, list_of_mins: list) -> list:
    closest_mins = []

    for peak_idx in list_of_peaks:
        # Filter the mins for those less than and greater than the current peak
        less_than_peak = [m for m in list_of_mins if m < peak_idx]
        greater_than_peak = [m for m in list_of_mins if m > peak_idx]

        # Find the closest less and greater mins
        min_less = max(less_than_peak) if less_than_peak else None
        min_greater = min(greater_than_peak) if greater_than_peak else None

        # Append the result as a tuple
        closest_mins.append((min_less, min_greater))

    return closest_mins
