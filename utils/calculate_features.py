def calculate_keystroke_features(key_press_times, key_release_times):
    print(f"Key Press Times: {key_press_times}")
    print(f"Key Release Times: {key_release_times}")
    """
    This function calculates key intervals, durations, hold times, total typing time, 
    and typing speed for keystroke dynamics based on key press and release timestamps.

    Args:
        key_press_times (list of float): Timestamps when keys were pressed.
        key_release_times (list of float): Timestamps when keys were released.

    Returns:
        dict: A dictionary containing calculated features.
    """
    
    # Ensure the press and release times match
    if len(key_press_times) != len(key_release_times):
        raise ValueError("Mismatch between key press and release times")

    # Calculate press-press intervals
    press_press_intervals = [
        key_press_times[i] - key_press_times[i - 1]
        for i in range(1, len(key_press_times))
    ]

    # Calculate release-press intervals
    release_press_intervals = [
        key_press_times[i] - key_release_times[i - 1]
        for i in range(1, len(key_press_times))
    ]

    # Calculate hold times (the duration each key was held down)
    hold_times = [
        key_release_times[i] - key_press_times[i]
        for i in range(len(key_press_times))
    ]
    press_to_release_ratio_mean = (
        sum(release_press_intervals) / sum(press_press_intervals)
        if sum(press_press_intervals) != 0 else 0
    )

    # Calculate total typing time
    total_typing_time = key_release_times[-1] - key_press_times[0]

    # Calculate typing speed (characters per second) based on number of key presses
    total_characters = len(key_press_times)
    typing_speed_cps = total_characters / \
        (total_typing_time / 1000)  # Convert ms to seconds
    
    # Return all calculated features
    return {
        'press_press_intervals': press_press_intervals,
        'release_press_intervals': release_press_intervals,
        'hold_times': hold_times,
        'total_typing_time': total_typing_time,
        'typing_speed_cps': typing_speed_cps,  # Characters per second
        'press_to_release_ratio_mean': press_to_release_ratio_mean,
    }
