archs = {
    "dblp": [
        ['AP', 'APT', 'APVP', 'APAPA', 'APTPA', 'APTPT', 'APTPV', 'APVPA', 'APVPV', 'APAPAP', 'APAPTP', 'APAPVP',
         'APTPTP', 'APTPVP', 'APVPTP', 'APAPAPV', 'APAPTPA', 'APAPTPV', 'APAPVPA', 'APAPVPT', 'APAPVPV', 'APTPAPA',
         'APTPAPT', 'APTPTPT', 'APTPVPV', 'APVPAPT', 'APVPTPA', 'APVPVPA', 'APVPVPT', 'APVPVPV'], []],
    "imdb": [
        ['M', 'MA', 'MK', 'MAM', 'MDM', 'MKM', 'MAMK', 'MDMK', 'MKMD', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM', 'MAMAMK',
         'MAMDMA', 'MAMDMK', 'MAMKMD', 'MAMKMK', 'MDMAMA', 'MDMAMD', 'MDMDMA', 'MDMDMK', 'MDMKMA', 'MKMAMA', 'MKMAMD',
         'MKMAMK', 'MKMDMA', 'MKMDMK', 'MKMKMD', 'MKMKMK'], []],
    "acm": [
        ['PPP', 'PAPP', 'PCPA', 'PCPP', 'PPPC', 'PPPP', 'PAPAP', 'PAPPP', 'PCPAP', 'PCPPA', 'PPAPA', 'PPAPC', 'PPAPP',
         'PAPAPA', 'PAPCPA', 'PAPPAP', 'PAPPCP', 'PAPPPP', 'PCPAPP', 'PCPCPP', 'PCPPAP', 'PCPPPP', 'PPAPAP', 'PPAPCP',
         'PPAPPA', 'PPAPPP', 'PPCPAP', 'PPPAPA', 'PPPCPA', 'PPPPPP'], []],
    "amazon_lp_han": [
        ['PCPPUP', 'PPPPUP', 'PPUPUP', 'PUPCCP', 'PUPCPP'], []],
    "amazon_lp_han2": [
        ['PCPPUP', 'PPPPUP', 'PPUPPP', 'PPUPUP', 'PUPPCP'], []],
    "amazon_lp_han3": [
        ['PUPP', 'PCPCP', 'PPCCCP', 'PPCPUP', 'PPUPUP'], []],
    # hand-engineered baseline (what an expert would define with NO discovery):
    # co-purchase, same-category, co-reviewed, 2-hop co-purchase, sibling-category.
    # 5 paths to match the discovered amazon_lp_han count, so the ablation
    # "does AUTOMATIC path discovery beat manual paths?" compares path QUALITY at
    # an equal path budget -- see random_paths.py for the matching random row.
    "amazon_lp_manual": [
        ['PP', 'PCP', 'PUP', 'PPP', 'PCCP'], []],

    #"amazon_lp_auto_0": [['PPUPP', 'PPPCPP', 'PUPCCP', 'PPPUPP'], []],
    "amazon_lp_auto_0": [['PPPUP', 'PPUPCP', 'PPUPP', 'PPUPPP', 'PPUPUP', 'PUP', 'PUPCCP', 'PUPCP', 'PUPCPP', 'PUPP', 'PUPPP', 'PUPPPP', 'PUPPUP', 'PUPUP', 'PUPUPP'], []],

    "amazon_lp_auto_1": [['PPUPCP', 'PPUPUP', 'PUPCPP', 'PUPPUP'], []], # (duplicate 1 x5)
    "amazon_lp_auto_2": [['PUPUP', 'PPUPCP', 'PPUPUP', 'PUPCPP'], []],  # (duplicate 2 x4)
    "amazon_lp_auto_3": [['PPUPCP', 'PPUPUP', 'PUPCPP', 'PUPUPP'], []], # (duplicate 3 x4)
    "amazon_lp_auto_4": [['PUPUP', 'PPUPCP', 'PPUPUP', 'PUPPUP'], []],  # (duplicate 4 x3)
    "amazon_lp_auto_5": [['PPUPCP', 'PPUPUP', 'PUPPUP', 'PUPUPP'], []], # (duplicate 5 x3)
    "amazon_lp_auto_6": [['PPUPP', 'PUPUP', 'PPUPCP', 'PUPPUP'], []],
    "amazon_lp_auto_7": [['PPUPP', 'PPUPCP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_auto_8": [['PUPUP', 'PPPCPP', 'PPUPCP', 'PPUPUP'], []],
    "amazon_lp_auto_9": [['PUPUP', 'PPUPCP', 'PPUPUP', 'PUPUPP'], []],
    "amazon_lp_auto_10": [['PPUPP', 'PPUPUP', 'PUPCPP', 'PUPPUP'], []],
    "amazon_lp_auto_11": [['PUPUP', 'PPPCPP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_auto_12": [['PPPCPP', 'PPUPCP', 'PPUPUP', 'PUPUPP'], []],
    "amazon_lp_auto_13": [['PPUPCP', 'PPUPUP', 'PUPCCP', 'PUPUPP'], []],
    "amazon_lp_auto_14": [['PPUPCP', 'PUPCPP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_15": [['PPPCPP', 'PPUPCP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_auto_16": [['PPPUPP', 'PPUPCP', 'PUPCPP', 'PUPPUP'], []],

    "amazon_lp_auto_17": [['PUP', 'PPPUP', 'PUPCP', 'PUPUP', 'PUPPUP', 'PUPUPP'], []],    # (duplicate 1 x3)
    "amazon_lp_auto_18": [['PUPCP', 'PUPPP', 'PUPUP', 'PUPCCP', 'PUPPUP', 'PUPUPP'], []], # (duplicate 2 x2)
    "amazon_lp_auto_19": [['PPUPP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPCPP', 'PUPPUP'], []], # (duplicate 3 x2)
    "amazon_lp_auto_20": [['PUP', 'PPUPP', 'PUPCP', 'PUPUP', 'PPUPCP', 'PUPPUP'], []],
    "amazon_lp_auto_21": [['PUP', 'PPPUP', 'PUPCP', 'PUPUP', 'PUPCCP', 'PUPPUP'], []],
    "amazon_lp_auto_22": [['PUP', 'PPUPP', 'PUPCP', 'PUPUP', 'PUPCPP', 'PUPPUP'], []],
    "amazon_lp_auto_23": [['PUP', 'PUPCP', 'PUPUP', 'PUPCPP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_24": [['PUP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPCPP', 'PUPUPP'], []],
    "amazon_lp_auto_25": [['PUP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_26": [['PUP', 'PUPCP', 'PUPUP', 'PUPCCP', 'PUPCPP', 'PUPUPP'], []],
    "amazon_lp_auto_27": [['PUP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPCPP', 'PUPPUP'], []],
    "amazon_lp_auto_28": [['PUP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPCCP', 'PUPUPP'], []],
    "amazon_lp_auto_29": [['PUP', 'PUPCP', 'PUPUP', 'PUPCCP', 'PUPCPP', 'PUPPUP'], []],
    "amazon_lp_auto_30": [['PUPP', 'PUPCP', 'PUPUP', 'PPUPCP', 'PUPCCP', 'PUPUPP'], []],
    "amazon_lp_auto_31": [['PUP', 'PUPCP', 'PPUPUP', 'PUPCPP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_32": [['PUPP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPCPP', 'PUPPUP'], []],
    "amazon_lp_auto_33": [['PPPUP', 'PUPCP', 'PUPUP', 'PPUPUP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_34": [['PPPUP', 'PUPCP', 'PUPUP', 'PUPCPP', 'PUPPPP', 'PUPUPP'], []],
    "amazon_lp_auto_35": [['PPPUP', 'PUPCP', 'PUPUP', 'PUPCPP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_36": [['PPUPP', 'PUPCP', 'PUPUP', 'PUPCPP', 'PUPPPP', 'PUPPUP'], []],
    "amazon_lp_auto_37": [['PPUPP', 'PUPCP', 'PUPUP', 'PUPCCP', 'PUPCPP', 'PUPUPP'], []],
    "amazon_lp_auto_38": [['PPUPP', 'PUPCP', 'PUPUP', 'PUPCPP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_39": [['PUPCP', 'PUPUP', 'PPUPPP', 'PUPCCP', 'PUPCPP', 'PUPUPP'], []],
    "amazon_lp_auto_40": [['PUPCP', 'PUPUP', 'PPUPUP', 'PUPCPP', 'PUPPUP', 'PUPUPP'], []],
    "amazon_lp_auto_41": [['PUPCP', 'PUPUP', 'PPUPUP', 'PUPCPP', 'PUPPPP', 'PUPUPP'], []],
    "amazon_lp_auto_42": [['PUPCP', 'PUPUP', 'PPUPUP', 'PUPCCP', 'PUPPPP', 'PUPUPP'], []],

    "amazon_lp_manual_0": [['PP', 'PCP', 'PPP', 'PUP', 'PCPP', 'PPCP', 'PPPP', 'PPUP', 'PUPP', 'PCPCP', 'PCPUP', 'PUPCP', 'PUPUP', 'PCPPUP', 'PPUPCP', 'PUPCPP', 'PPUPUP', 'PUPPUP'], []],

    "amazon_lp_manual_1": [['PP', 'PUP', 'PCP', 'PPP'], []],
    "amazon_lp_manual_2": [['PUP', 'PUPUP', 'PPUP', 'PUPP'], []],
    "amazon_lp_manual_3": [['PCP', 'PCPCP', 'PPCP', 'PCPP'], []],
    "amazon_lp_manual_4": [['PUP', 'PCP', 'PUPCP', 'PCPUP'], []],
    "amazon_lp_manual_5": [['PPP', 'PPPP', 'PPUP', 'PUPP'], []],
    "amazon_lp_manual_6": [['PUPCP', 'PPUPCP', 'PUPCPP', 'PCPPUP'], []],
    "amazon_lp_manual_7": [['PUPUP', 'PUPCP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_manual_8": [['PUPUP', 'PCPCP', 'PPUPCP', 'PUPPUP'], []],
}
