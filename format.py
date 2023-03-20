import csv

def convert_to_full(filepath, type):
    # This will convert the single letters to full words for an easier read
    
    options = {

                    "poisonous/edible": {"e": {"name": "edible", "number": 0}, "p": {"name": "poisonous", "number": 1}},
                    "cap-shape": {"b": {"name": "bell", "number": 0}, "c": {"name": "conical", "number": 1}, "x": {"name": "convex", "number": 2}, "f": {"name": "flat", "number": 3}, "k": {"name": "knobbed", "number": 4}, "s": {"name": "sunken", "number": 5}},
                    "cap-surface": {"f": {"name": "fibrous", "number": 0}, "g": {"name": "grooves", "number": 1}, "y": {"name": "scaly", "number": 2}, "s": {"name": "smooth", "number": 3}},
                    "cap-color": {"n": {"name": "brown", "number": 0}, "b": {"name": "buff", "number": 1}, "c": {"name": "cinnamon", "number": 2}, "g": {"name": "gray", "number": 3}, "r": {"name": "green", "number": 4}, "p": {"name": "pink", "number": 5}, "u": {"name": "purple", "number": 6}, "e": {"name": "red", "number": 7}, "w": {"name": "white", "number": 8}, "y": {"name": "yellow", "number": 9}},
                    "bruises?": {"f": {"name": "no bruises", "number": 0}, "t": {"name": "bruises", "number": 1}},
                    "odor": {"a": {"name": "almonds", "number": 0}, "l": {"name": "anise", "number": 1}, "c": {"name": "creosote", "number": 2}, "y": {"name": "fishy", "number": 3}, "f": {"name": "foul", "number": 4}, "m": {"name": "musty", "number": 5}, "n": {"name": "None", "number": 6}, "p": {"name": "pungent", "number": 7}, "s": {"name": "spicy", "number": 8}},
                    "gill-attachment": {"a": {"name": "attached", "number": 0}, "d": {"name": "descending", "number": 1}, "f": {"name": "free", "number": 2}, "n": {"name": "notched", "number": 3}},
                    "gill-spacing": {"c": {"name": "close", "number": 0}, "w": {"name": "crowded", "number": 1}, "d": {"name": "distant", "number": 2}},
                    "gill-size": {"b": {"name": "broad", "number": 0}, "n": {"name": "narrow", "number": 1}},
                    "gill-color": {"k": {"name": "black", "number": 0}, "n": {"name": "brown", "number": 1}, "b": {"name": "buff", "number": 2}, "h": {"name": "chocolate", "number": 3}, "g": {"name": "gray", "number": 4}, "r": {"name": "green", "number": 5}, "o": {"name": "orange", "number": 6}, "p": {"name": "pink", "number": 7}, "u": {"name": "purple", "number": 8}, "e": {"name": "red", "number": 9}, "w": {"name": "white", "number": 10}, "y": {"name": "yellow", "number": 11}},
                    "stalk-shape": {"e": {"name": "enlarging", "number": 0}, "t": {"name": "tapering", "number": 1}},
                    "stalk-root": {"b": {"name": "bulbous", "number": 0}, "c": {"name": "club", "number": 1}, "u": {"name": "cup", "number": 2}, "e": {"name": "equal", "number": 3}, "z": {"name": "rhizomorphs", "number": 4}, "r": {"name": "rooted", "number": 5}, "?": {"name": "missing", "number": 6}},
                    "stalk-surface-above-ring": {"f": {"name": "fibrous", "number": 0}, "y": {"name": "scaly", "number": 1}, "k": {"name": "silky", "number": 2}, "s": {"name": "smooth", "number": 3}},
                    "stalk-surface-below-ring": {"f": {"name": "fibrous", "number": 0}, "y": {"name": "scaly", "number": 1}, "k": {"name": "silky", "number": 2}, "s": {"name": "smooth", "number": 3}},
                    "stalk-color-above-ring": {"n": {"name": "brown", "number": 0}, "b": {"name": "buff", "number": 1}, "c": {"name": "cinnamon", "number": 2}, "g": {"name": "gray", "number": 3}, "o": {"name": "orange", "number": 4}, "p": {"name": "pink", "number": 5}, "e": {"name": "red", "number": 6}, "w": {"name": "white", "number": 7}, "y": {"name": "yellow", "number": 8}},
                    "stalk-color-below-ring": {"n": {"name": "brown", "number": 0}, "b": {"name": "buff", "number": 1}, "c": {"name": "cinnamon", "number": 2}, "g": {"name": "gray", "number": 3}, "o": {"name": "orange", "number": 4}, "p": {"name": "pink", "number": 5}, "e": {"name": "red", "number": 6}, "w": {"name": "white", "number": 7}, "y": {"name": "yellow", "number": 8}},
                    "veil-type": {"p": {"name": "partial", "number": 0}, "u": {"name": "universal", "number": 1}},
                    "veil-color": {"n": {"name": "brown", "number": 0}, "o": {"name": "orange", "number": 1}, "w": {"name": "white", "number": 2}, "y": {"name": "yellow", "number": 3}},
                    "ring-number": {"n": {"name": "none", "number": 0}, "o": {"name": "one", "number": 1}, "t": {"name": "two", "number": 2}},
                    "ring-type": {"c": {"name": "cobwebby", "number": 0}, "e": {"name": "evanescent", "number": 1}, "f": {"name": "flaring", "number": 2}, "l": {"name": "large", "number": 3}, "n": {"name": "none", "number": 4}, "p": {"name": "pendant", "number": 5}, "s": {"name": "sheathing", "number": 6}, "z": {"name": "zone", "number": 7}},
                    "spore-print-color": {"k": {"name": "black", "number": 0}, "n": {"name": "brown", "number": 1}, "b": {"name": "buff", "number": 2}, "h": {"name": "chocolate", "number": 3}, "r": {"name": "green", "number": 4}, "o": {"name": "orange", "number": 5}, "u": {"name": "purple", "number": 6}, "w": {"name": "white", "number": 7}, "y": {"name": "yellow", "number": 8}},
                    "population": {"a": {"name": "abundant", "number": 0}, "c": {"name": "clustered", "number": 1}, "n": {"name": "numerous", "number": 2}, "s": {"name": "scattered", "number": 3}, "v": {"name": "several", "number": 4}, "y": {"name": "solitary", "number": 5}},
                    "habitat": {"g": {"name": "grasses", "number": 0}, "l": {"name": "leaves", "number": 1}, "m": {"name": "meadows", "number": 2}, "p": {"name": "paths", "number": 3}, "u": {"name": "urban", "number": 4}, "w": {"name": "waste", "number": 5}, "d": {"name": "woods", "number": 6}}
                }

    file = open(filepath, 'r')
    lines = file.readlines()

    full_data = {}
    row_count = 0
    char_count = 0

    for line in lines:

        row_data = {
                        row_count: {
                                    "poisonous/edible": None,
                                    "cap-shape": None,
                                    "cap-surface": None,
                                    "cap-color": None,
                                    "bruises?": None,
                                    "odor": None,
                                    "gill-attachment": None,
                                    "gill-spacing": None,
                                    "gill-size": None,
                                    "gill-color": None,
                                    "stalk-shape": None,
                                    "stalk-root": None,
                                    "stalk-surface-above-ring": None,
                                    "stalk-surface-below-ring": None,
                                    "stalk-color-above-ring": None,
                                    "veil-type": None,
                                    "veil-color": None,
                                    "ring-number": None,
                                    "ring-type": None,
                                    "spore-print-color": None,
                                    "population": None,
                                    "habitat": None
                            }
                    }
        
        for attribute in options:
            for key in options[attribute]:
                if line[char_count] == key:
                    row_data[row_count][attribute] = options[attribute][key][type]
                    char_count += 2
                    break
            
        row_count += 1
        char_count = 0
        full_data.update(row_data)

    return full_data

if __name__ == '__main__':
    
    data_numbers = convert_to_full("agaricus-lepiota.data", "number")

    columns = ["poisonous/edible","cap-shape", "cap-surface",
               "cap-color", "bruises?", "odor", "gill-attachment",
               "gill-spacing", "gill-size", "gill-color", 
               "stalk-shape", "stalk-root", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring",
               "stalk-color-below-ring", "veil-type", "veil-color",
               "ring-number", "ring-type", "spore-print-color",
               "population", "habitat"]

    with open('mush_data_numbers.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames= columns)
        writer.writeheader()
        for key in data_numbers:
            writer.writerow(data_numbers[key])

    data_names = convert_to_full("agaricus-lepiota.data", "name")

    columns = ["poisonous/edible","cap-shape", "cap-surface",
               "cap-color", "bruises?", "odor", "gill-attachment",
               "gill-spacing", "gill-size", "gill-color", 
               "stalk-shape", "stalk-root", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring",
               "stalk-color-below-ring", "veil-type", "veil-color",
               "ring-number", "ring-type", "spore-print-color",
               "population", "habitat"]

    with open('mush_data_names.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames= columns)
        writer.writeheader()
        for key in data_names:
            writer.writerow(data_names[key])
  