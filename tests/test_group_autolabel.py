from nerd.sheetbuilder.groups import GroupSet, ReactionGroup


def test_explicit_construct_is_not_appended_again():
    groups = GroupSet()
    groups.label_template = '{temperature}_{replicate}_{construct}'
    groups.groups = [
        ReactionGroup(key=('buffer', 'rep1', '80', 'other')),
        ReactionGroup(key=('buffer', 'rep2', '80', 'hsp17wt')),
    ]
    groups._autolabel()
    assert [g.label for g in groups.groups] == ['80_rep1_other', '80_rep2_hsp17wt']


def test_default_template_keeps_variant_suffix():
    groups = GroupSet()
    groups.groups = [
        ReactionGroup(key=('buffer', '1', '80', 'wt')),
        ReactionGroup(key=('buffer', '2', '80', 'mutant')),
    ]
    groups._autolabel()
    assert [g.label for g in groups.groups] == ['80_1', '80_2_mutant']
