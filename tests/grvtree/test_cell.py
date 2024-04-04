import io

import pytest
from grvtree import GRVCell, GRVCellCompareScore

from nltk.tree import Tree
import nltk_tree_ext.patch


class TestGRVCell:
    RAW_TREES_WITH_GDV = (
        (
            "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))))",
            (
                GRVCell(2, "NP", "PRP", "My"),
                GRVCell(-1, "S", "NN", "daughter"),
                GRVCell(1, "VP", "VBD", "broke"),
                GRVCell(2, "NP", "DET", "the"),
                GRVCell(0, "NP", "JJ", "red"),
                GRVCell(-1, "NP", "NN", "toy"),
                GRVCell(1, "PP", "IN", "with"),
                GRVCell(1, "NP", "DET", "a"),
                GRVCell(0, "S", "NN", "hammer"),
            ),
        ),
        (
            "(IP-MAT (ADJI ありがとう) (VB2 ござい) (AX ます))",
            (
                GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(0, "IP-MAT", "AX", "ます"),
            ),
        ),
        (
            # 54_aozora_Kajii-1925;JP
            """
(IP-MAT (NP-SBJ *speaker*)
        (PP (IP-ADV (PP (NP (PRO そこ))
                        (P は))
                    (NP-SBJ *)
                    (ADVP (ADV 決して))
                    (NP-PRD (IP-REL (NP-SBJ *T*)
                                    (ADJN 立派)
                                    (AX な))
                            (N 店))
                    (AX で)
                    (P は)
                    (NEG なかっ)
                    (AXD た)
                    (FN の)
                    (AX だ))
            (P が))
        (CONJ *)
        (PU 、)
        (PP (NP (PP (NP (N 果物屋)
                        (N 固有))
                    (P の))
                (N 美しさ))
            (P が))
        (NP-OB1 *が*)
        (ADVP (ADV 最も)
            (ADJN 露骨)
            (AX に))
        (VB 感ぜ)
        (VB2 られ)
        (AXD た)
        (PU 。))
            """,
            (
                GRVCell(
                    form="*speaker*",
                    lex_cat="NP-SBJ",
                    height_diff=1,
                    phrase_cat="IP-MAT",
                ),
                GRVCell(form="そこ", lex_cat="NP☆PRO", height_diff=3, phrase_cat="PP"),
                GRVCell(form="は", lex_cat="P", height_diff=-1, phrase_cat="IP-ADV"),
                GRVCell(form="*", lex_cat="NP-SBJ", height_diff=0, phrase_cat="IP-ADV"),
                GRVCell(
                    form="決して",
                    lex_cat="ADVP☆ADV",
                    height_diff=0,
                    phrase_cat="IP-ADV",
                ),
                GRVCell(
                    form="*T*", lex_cat="NP-SBJ", height_diff=2, phrase_cat="IP-REL"
                ),
                GRVCell(
                    form="立派", lex_cat="ADJN", height_diff=0, phrase_cat="IP-REL"
                ),
                GRVCell(form="な", lex_cat="AX", height_diff=-1, phrase_cat="NP-PRD"),
                GRVCell(form="店", lex_cat="N", height_diff=-1, phrase_cat="IP-ADV"),
                GRVCell(form="で", lex_cat="AX", height_diff=0, phrase_cat="IP-ADV"),
                GRVCell(form="は", lex_cat="P", height_diff=0, phrase_cat="IP-ADV"),
                GRVCell(
                    form="なかっ", lex_cat="NEG", height_diff=0, phrase_cat="IP-ADV"
                ),
                GRVCell(form="た", lex_cat="AXD", height_diff=0, phrase_cat="IP-ADV"),
                GRVCell(form="の", lex_cat="FN", height_diff=0, phrase_cat="IP-ADV"),
                GRVCell(form="だ", lex_cat="AX", height_diff=-1, phrase_cat="PP"),
                GRVCell(form="が", lex_cat="P", height_diff=-1, phrase_cat="IP-MAT"),
                GRVCell(form="*", lex_cat="CONJ", height_diff=0, phrase_cat="IP-MAT"),
                GRVCell(form="、", lex_cat="PU", height_diff=0, phrase_cat="IP-MAT"),
                GRVCell(form="果物屋", lex_cat="N", height_diff=4, phrase_cat="NP"),
                GRVCell(form="固有", lex_cat="N", height_diff=-1, phrase_cat="PP"),
                GRVCell(form="の", lex_cat="P", height_diff=-1, phrase_cat="NP"),
                GRVCell(form="美しさ", lex_cat="N", height_diff=-1, phrase_cat="PP"),
                GRVCell(form="が", lex_cat="P", height_diff=-1, phrase_cat="IP-MAT"),
                GRVCell(
                    form="*が*", lex_cat="NP-OB1", height_diff=0, phrase_cat="IP-MAT"
                ),
                GRVCell(form="最も", lex_cat="ADV", height_diff=1, phrase_cat="ADVP"),
                GRVCell(form="露骨", lex_cat="ADJN", height_diff=0, phrase_cat="ADVP"),
                GRVCell(form="に", lex_cat="AX", height_diff=-1, phrase_cat="IP-MAT"),
                GRVCell(form="感ぜ", lex_cat="VB", height_diff=0, phrase_cat="IP-MAT"),
                GRVCell(form="られ", lex_cat="VB2", height_diff=0, phrase_cat="IP-MAT"),
                GRVCell(form="た", lex_cat="AXD", height_diff=0, phrase_cat="IP-MAT"),
                GRVCell(form="。", lex_cat="PU", height_diff=0, phrase_cat="IP-MAT"),
            ),
        ),
    )

    @pytest.mark.parametrize("tree_raw, result", RAW_TREES_WITH_GDV)
    def test_encode_GRV(self, tree_raw: str, result):
        tree_parsed = Tree.fromstring(tree_raw).merge_nonterminal_unary_nodes(
            lambda a, b: f"{a}☆{b}"
        )
        tree_encoded = tuple(GRVCell.encode_nltk_tree(tree_parsed))

        assert tree_encoded == result

    @pytest.mark.parametrize("tree_raw, cells", RAW_TREES_WITH_GDV)
    def test_decode_GRV(self, tree_raw: str, cells):
        tree_decoded = GRVCell.decode_as_nltk_tree(cells)
        tree_raw_parsed = Tree.fromstring(tree_raw).merge_nonterminal_unary_nodes(
            lambda a, b: f"{a}☆{b}"
        )

        print(f"DECODED: {tree_decoded}")
        print(f"SHOULD BE: {tree_raw_parsed}")

        assert tree_decoded == tree_raw_parsed

    GRV_CELL_SEQ = (
        (
            (
                GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(0, "IP-MAT", "AX", "ます"),
            ),
            (
                GRVCell(233, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(234324, "adfaf", "AX", "ます"),
            ),
            True,
        ),
        (
            (
                GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(0, "IP-MAT", "AX", "ます"),
            ),
            (
                GRVCell(233, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(234324, "adfaf", "AX", "あああ"),
            ),
            False,
        ),
        (tuple(), tuple(), True),
        (
            (
                GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(0, "IP-MAT", "AX", "ます"),
            ),
            (GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),),
            False,
        ),
    )

    @pytest.mark.parametrize("this, other, expected", GRV_CELL_SEQ)
    def test_seq_equal(self, this, other, expected):
        assert GRVCell.seq_equal(this, other) == expected

    GRV_CELL_SEQ_SCORE = (
        (
            (
                GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(0, "IP-MAT", "AX", "ます"),
            ),
            (
                GRVCell(233, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(234324, "adfaf", "AX", "ます"),
            ),
            GRVCellCompareScore(
                length_this=3,
                length_other=3,
                first_height_diff_match=False,
                matched_height_diff=1,
                matched_phrase_cat=2,
                matched_lex_cat=3,
                matched_form=3,
            ),
        ),
        (
            tuple(),
            tuple(),
            GRVCellCompareScore(
                length_this=0,
                length_other=0,
                first_height_diff_match=False,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=0,
                matched_form=0,
            ),
        ),
        (
            (
                GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),
                GRVCell(0, "IP-MAT", "VB2", "ござい"),
                GRVCell(0, "IP-MAT", "AX", "ます"),
            ),
            (GRVCell(1, "IP-MAT", "ADJI", "ありがとう"),),
            GRVCellCompareScore(
                length_this=3,
                length_other=1,
                first_height_diff_match=False,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=1,
                matched_form=1,
            ),
        ),
    )

    @pytest.mark.parametrize("this, other, expected", GRV_CELL_SEQ_SCORE)
    def test_seq_compare(self, this, other, expected):
        result = GRVCell.seq_compare(this, other)
        assert result == expected


class TestGRVCellCompareScore:
    SCORE_HEIGHT_DIFF = (
        (
            GRVCellCompareScore(
                length_this=3,
                length_other=3,
                first_height_diff_match=False,
                matched_height_diff=1,
                matched_phrase_cat=2,
                matched_lex_cat=3,
                matched_form=3,
            ),
            1,
            1 / 2,
        ),
        (
            GRVCellCompareScore(
                length_this=0,
                length_other=0,
                first_height_diff_match=False,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=0,
                matched_form=0,
            ),
            1,
            1,
        ),
        (
            GRVCellCompareScore(
                length_this=3,
                length_other=1,
                first_height_diff_match=False,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=1,
                matched_form=1,
            ),
            0,
            0,
        ),
        (
            GRVCellCompareScore(
                length_this=3,
                length_other=2,
                first_height_diff_match=True,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=1,
                matched_form=1,
            ),
            0,
            1 / 2,
        ),
    )

    @pytest.mark.parametrize("score, prec_rel, prec_abs", SCORE_HEIGHT_DIFF)
    def test_precision_height_diff(self, score, prec_rel, prec_abs):
        assert score.precision_height_diff(relative=True) == prec_rel
        assert score.precision_height_diff(relative=False) == prec_abs


    SCORE_PHRASE_CAT = (
        (
            GRVCellCompareScore(
                length_this=3,
                length_other=3,
                first_height_diff_match=False,
                matched_height_diff=1,
                matched_phrase_cat=2,
                matched_lex_cat=3,
                matched_form=3,
            ),
            1 / 2,
        ),
        (
            GRVCellCompareScore(
                length_this=0,
                length_other=0,
                first_height_diff_match=False,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=0,
                matched_form=0,
            ),
            1,
        ),
        (
            GRVCellCompareScore(
                length_this=3,
                length_other=1,
                first_height_diff_match=True,
                matched_height_diff=0,
                matched_phrase_cat=0,
                matched_lex_cat=1,
                matched_form=1,
            ),
            1,
        ),
    )