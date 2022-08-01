LANGS = "fr cs de fi lv et ro hi tr gu".split()

if __name__ == "__main__":
    cmds = ""
    MODEL = "/path/to/avg11_15.pt"
    BATCH_SIZE = 128
    BEAM = 5
    for lang in LANGS:
        cmd = "- name: grads_en2{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/aml/multi-node/two-stage/wmt10/32GPU/A100/x2x/Grads/generate_grads.sh {} {}\n".format(
            lang, lang, "en-{}".format(lang), MODEL)
        cmds += cmd
    for lang in LANGS:
        cmd = "- name: grads_{}2en\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/aml/multi-node/two-stage/wmt10/32GPU/A100/x2x/Grads/generate_grads.sh {} {}\n".format(
            lang, lang, "{}-en".format(lang), MODEL)
        cmds += cmd
    print(cmds)
