def parse_result(result):
    if len(result) > 2:
            id_img, bool_zoom, mina, maxa, minb, maxb = result # parsed line from txt file
            mina = int(mina)
            maxa = int(maxa)
            minb = int(minb)
            maxb = int(maxb)
    else:
        id_img, bool_zoom = result
        mina = minb = maxa = maxb = None
    
    return id_img, bool_zoom, mina, maxa, minb, maxb