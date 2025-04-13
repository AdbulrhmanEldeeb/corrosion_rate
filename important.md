    classes(mm/yr)
    
    if val < 0.0508:
        return 0  # A
    elif val < 0.508:
        return 1  # B
    elif val <= 1.27:
        return 2  # C
    else:
        return 3  # D

make staless stell default value in ui 
add explaination in the ui for concentration that should be a numeric value , percentage like 70