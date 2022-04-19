def apply_pixel_mask(arr, threshold=0.7):
    newArr = arr.copy()
    threshold *= arr.max()
    newArr[newArr < threshold] = 0
    newArr[newArr >= threshold] = 1
    return newArr

def normalize(arr):
    """Normalize an array to int8 (0-255)"""

    min = arr.min()
    max = arr.max()
    if max == 0 or min == max:
        return arr

    arr = arr - min
    return arr * 255 / max
