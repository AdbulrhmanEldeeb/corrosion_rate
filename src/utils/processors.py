import re
import numpy as np 


def clean_condition_text(text):

    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-z0-9%.\- ]+", "", text)
    return text


def extract_features_from_comment(comment):
    features = {}

    # Flags (binary)
    features["is_glacial"] = int(bool(re.search(r"glacial", comment, re.IGNORECASE)))
    features["aerated"] = int(
        bool(
            re.search(
                r"aerated|air saturated|air sparge|strong aeration",
                comment,
                re.IGNORECASE,
            )
        )
    )
    features["unaerated"] = int(
        bool(
            re.search(
                r"no air|unaerated|no aeration|nitrogen saturated",
                comment,
                re.IGNORECASE,
            )
        )
    )
    features["agitation_moderate"] = int(
        bool(re.search(r"slight to moderate agitation", comment, re.IGNORECASE))
    )
    features["agitation_static"] = int(
        bool(re.search(r"static", comment, re.IGNORECASE))
    )

    features["mill_annealed"] = int(
        bool(re.search(r"mill annealed", comment, re.IGNORECASE))
    )
    features["heat_treated"] = int(
        bool(re.search(r"heat treated|annealed", comment, re.IGNORECASE))
    )
    features["cast_specimen"] = int(
        bool(re.search(r"cast specimen", comment, re.IGNORECASE))
    )

    features["lab_test"] = int(bool(re.search(r"lab test", comment, re.IGNORECASE)))
    features["plant_test"] = int(bool(re.search(r"plant test", comment, re.IGNORECASE)))
    features["evaporator"] = int(bool(re.search(r"evaporator", comment, re.IGNORECASE)))
    features["diaphragm_cell"] = int(
        bool(re.search(r"diaphragm cell", comment, re.IGNORECASE))
    )
    features["mercury_cell"] = int(
        bool(re.search(r"mercury cell", comment, re.IGNORECASE))
    )

    # Helper to extract numerical value
    def extract_value(pattern):
        match = re.search(pattern, comment, re.IGNORECASE)
        return float(match.group(1)) if match else np.nan

    # Numerical
    features["ppm_cu2"] = extract_value(r"(\d+(?:\.\d+)?)\s*ppm\s*cu2")
    features["ppm_cl"] = extract_value(r"(\d+(?:\.\d+)?)\s*ppm\s*cl")
    features["gpl_fe"] = extract_value(r"(\d+(?:\.\d+)?)\s*g/l\s*fe")
    features["cu_so4_%"] = extract_value(r"(\d+(?:\.\d+)?)%\s*cu")
    features["na_cl_%"] = extract_value(r"(\d+(?:\.\d+)?)%\s*nacl")
    features["hno3_%"] = extract_value(r"(\d+(?:\.\d+)?)%\s*hno3")
    features["pH"] = extract_value(r"pH\s*(\d+(?:\.\d+)?)")

    return features
