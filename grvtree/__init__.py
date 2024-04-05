from typing import Generic, TypeVar, Type, Callable, Any, cast
import itertools
import more_itertools
from collections.abc import Iterator, Sequence, Iterable
import dataclasses
from dataclasses import dataclass
from nltk.tree import Tree
import nltk_tree_ext.patch

F = TypeVar("F")
L = TypeVar("L")
P = TypeVar("P")
NODE = TypeVar("NODE")
LEAF = TypeVar("LEAF")


@dataclass(frozen=True, slots=True)
class GRVCellCompareScore:
    length_this: int
    length_other: int
    first_height_diff_match: bool
    """
    Whether the first height differences match.
    """

    matched_height_diff: int
    """
    The count of matched height differences.

    Notes
    -----
    The first cells are not counted.
    The last cells are not counted, either.
    """

    matched_phrase_cat: int
    """
    The count of matched phrase categories.

    Notes
    -----
    The last cells are not counted, either.
    """

    matched_lex_cat: int
    matched_form: int

    def lengths(
        self, cut_end: bool = True, relative: bool = True
    ):  # -> tuple[Any, Any]:# -> tuple[Any, Any]:
        return (
            self.length_this - (1 if cut_end else 0) - (1 if relative else 0),
            self.length_other - (1 if cut_end else 0) - (1 if relative else 0),
        )

    def precision_height_diff(self, relative: bool = True) -> float:
        len_this, len_other = self.lengths(cut_end=True, relative=relative)
        if len_this < 1 and len_other < 1:
            return 1
        else:
            return (
                self.matched_height_diff
                + (1 if not relative and self.first_height_diff_match else 0)
            ) / max(len_this, len_other)

    def precision_phrase_cat(self) -> float:
        if (max_len := max(self.lengths(cut_end=True, relative=False))) <= 1:
            return 1
        else:
            return self.matched_phrase_cat / (max_len - 1)

    def precision_lex_cat(self) -> float:
        return self.matched_lex_cat / max(self.lengths(cut_end=False, relative=False))

    def precision_form(self) -> float:
        return self.matched_form / max(self.lengths(cut_end=False, relative=False))

    def to_dict(self, relative: bool = True) -> dict[str, int | float]:
        return {
            **dataclasses.asdict(self),
            "precision_height_diff": self.precision_height_diff(relative=relative),
            "precision_phrase_cat": self.precision_phrase_cat(),
            "precision_lex_cat": self.precision_lex_cat(),
            "precision_form": self.precision_form(),
        }


@dataclass(frozen=True, slots=True)
class GRVCell(Generic[P, L, F]):
    """
    Represents a cell of a GRV-encoded tree.
    The GRV-encoding refers to the linearization of a tree structure by the way described by [1]_.

    References
    ----------
    .. [1] Gómez-Rodríguez, C., & Vilares, D. (2018). Constituent Parsing as Sequence Labeling. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1314–1324. https://doi.org/10.18653/v1/D18-1162

    """

    height_diff: int
    """
    The difference of the height of the current cell from the previous one.
    """

    phrase_cat: P
    """
    The lowest phrase category among those shared with the next branch.
    """

    lex_cat: L
    """
    The lexical category of this cell.
    """

    form: F
    """
    The lexical form of this cell.
    """

    @classmethod
    def seq_compare(
        cls: Type["GRVCell[P, L, F]"],
        this: "Iterable[GRVCell[P, L, F]]",
        other: "Iterable[GRVCell[P, L, F]]",
        eq_phrase: Callable[[P, P], bool] = lambda x, y: x == y,
        eq_lex: Callable[[L, L], bool] = lambda x, y: x == y,
        eq_form: Callable[[F, F], bool] = lambda x, y: x == y,
    ) -> GRVCellCompareScore:
        """
        Compare two sequences of cells.

        Arguments
        ---------
        relative
            If True, the absolute height difference is ignored.
        """
        this = tuple(this)
        other = tuple(other)

        first_height_diff_match: bool = False
        matched_height_diff: int = 0
        matched_phrase_cat: int = 0
        matched_lex_cat: int = 0
        matched_form: int = 0

        for tuple_this, tuple_other in itertools.zip_longest(
            more_itertools.mark_ends(this), more_itertools.mark_ends(other)
        ):
            match tuple_this, tuple_other:
                case None, None:
                    break
                case None, (is_begin, _, cell_other):
                    if is_begin:
                        first_height_diff_match = False
                    # No incr because the left side is None
                case (is_begin, _, cell_this), None:
                    if is_begin:
                        first_height_diff_match = False
                    # No incr because the left side is None
                case (is_begin, is_end_this, cell_this), (_, is_end_other, cell_other):
                    if not is_end_this and not is_end_other:
                        if is_begin:
                            # only the first height difference which is not an end is counted
                            first_height_diff_match = (
                                cell_this.height_diff == cell_other.height_diff
                            )
                        matched_height_diff += (
                            cell_this.height_diff == cell_other.height_diff
                        )
                        matched_phrase_cat += eq_phrase(
                            cell_this.phrase_cat, cell_other.phrase_cat
                        )
                    matched_lex_cat += eq_lex(cell_this.lex_cat, cell_other.lex_cat)
                    matched_form += eq_form(cell_this.form, cell_other.form)
                case _:
                    raise ValueError("Unexpected case")

        return GRVCellCompareScore(
            length_this=len(this),
            length_other=len(other),
            first_height_diff_match=first_height_diff_match,
            matched_height_diff=matched_height_diff,
            matched_phrase_cat=matched_phrase_cat,
            matched_lex_cat=matched_lex_cat,
            matched_form=matched_form,
        )

    @classmethod
    def seq_equal(
        cls: Type["GRVCell[P, L, F]"],
        this: "Iterable[GRVCell[P, L, F]]",
        other: "Iterable[GRVCell[P, L, F]]",
        eq_phrase: Callable[[P, P], bool] = lambda x, y: x == y,
        eq_lex: Callable[[L, L], bool] = lambda x, y: x == y,
        eq_form: Callable[[F, F], bool] = lambda x, y: x == y,
        relative: bool = True,
    ) -> bool:
        """
        Check if two sequences of cells are equal in the sense that they represent the same tree.

        Arguments
        ---------
        relative
            If True, the absolute height difference is ignored.
        """
        for is_begin, is_end, (cell_this, cell_other) in more_itertools.mark_ends(
            itertools.zip_longest(this, other)
        ):
            match cell_this, cell_other:
                case None, None:
                    return True
                case _, None:
                    return False
                case None, _:
                    return False
                case t, o:
                    if not all(
                        (
                            (is_begin and relative)
                            or is_end
                            or t.height_diff == o.height_diff,
                            is_end or eq_phrase(t.phrase_cat, o.phrase_cat),
                            eq_lex(t.lex_cat, o.lex_cat),
                            eq_form(t.form, o.form),
                        )
                    ):
                        return False
        # if the loop ends without returning, the sequences are empty
        return True

    @classmethod
    def encode_nltk_tree(
        cls: Type["GRVCell[NODE, NODE, LEAF]"], tree: Tree[NODE, LEAF]
    ) -> Iterator["GRVCell[NODE, NODE, LEAF]"]:
        """
        Encode `tree`.
        The he
        Absolute scale is adopted.

        Notes
        -----
        Unary branches (except for those at leaves) should be merged beforehand.
        Otherwise the encoding is undefined.
        """
        prev_height: int = 0

        for (
            _,
            is_end,
            ((cl_branch, cl_leaf), (nl_branch, nl_leaf)),
        ) in more_itertools.mark_ends(
            more_itertools.windowed(tree.iter_leaves_with_branches(), 2, (None, None))
        ):
            if cl_branch is None or cl_leaf is None:
                raise ValueError("No leaves found in the tree.")
            elif nl_branch is None or nl_leaf is None:
                # single leaf in the tree
                yield cls(
                    height_diff=len(cl_branch),
                    phrase_cat=cl_branch[0],
                    lex_cat=cl_branch[-1],
                    form=cl_leaf,
                )
                return  # successfully encoded
            else:
                match_idx = next(
                    (
                        count_common_ancestors
                        for count_common_ancestors, (
                            current_node,
                            next_node,
                        ) in enumerate(zip(cl_branch[:-1], nl_branch[:-1]))
                        if current_node != next_node
                    ),
                    # default value
                    min(len(cl_branch), len(nl_branch)) - 1,
                )
                yield cls(
                    height_diff=match_idx - prev_height,
                    phrase_cat=cl_branch[match_idx - 1],
                    lex_cat=cl_branch[-1],
                    form=cl_leaf,
                )
                if is_end:
                    yield cls(
                        height_diff=0,  # arbitrary
                        phrase_cat=nl_branch[0],  # arbitrary
                        lex_cat=nl_branch[-1],
                        form=nl_leaf,
                    )
                    return  # successfully encoded
                prev_height = match_idx

        # If no leaves are found raise an error
        raise ValueError("No leaves found in the tree.")

    @classmethod
    def decode_as_nltk_tree(
        cls: Type["GRVCell[NODE, NODE, LEAF]"],
        cells: Sequence["GRVCell[NODE, NODE, LEAF]"],
        relativize_init_height: bool = True,
    ) -> Tree[NODE, LEAF]:
        """
        Decode an encoded tree.
        Relative scale is assumed.

        Arguments
        ---------
        cells : list of :class:`GRVCell`

        relativize_init_height : bool, default True

        Notes
        -----
        * Collapsed unary nodes is to be expanded manually after the decoding.
        * The `height_diff` of the first cell represents the absolute startline of branching. The counting of the height begins from `1`.
        """
        # Initial cell
        initial_cell = cells[0]
        tree_pointer: list[Tree[NODE, LEAF]] = []

        # Adjust the height
        initial_height: int = initial_cell.height_diff
        if relativize_init_height:
            min_abs_height = min(
                itertools.accumulate(
                    (c.height_diff for c in cells[1:]),
                    lambda x, y: x + y,
                    initial=initial_height,
                )
            )
            initial_height += 1 - min_abs_height

        if initial_height < 2:
            tree_pointer.append(
                Tree.fromlist_as_unary(
                    (
                        initial_cell.phrase_cat,
                        initial_cell.lex_cat,
                    ),
                    [initial_cell.form],
                )
            )
        else:
            # (initial_height - 1) branches will be grown from now on.
            # The strategy is that we first grow (initial_height - 2) empty branches by loop, ...
            new_node: Tree = Tree(None, [])  # type: ignore
            tree_pointer = [new_node]
            for _ in range(initial_height - 2):
                child: Tree[NODE, LEAF] = Tree(None, [])  # type: ignore
                # insert `child` to the last pointed tree
                tree_pointer[-1].append(child)
                # add `child` to the pointer stack
                tree_pointer.append(child)

            # ... and separately create the latest branch according to the cell.
            terminal_subtree = Tree.fromlist_as_unary(
                (initial_cell.phrase_cat, initial_cell.lex_cat),
                (initial_cell.form,),
            )
            tree_pointer[-1].append(terminal_subtree)  # type: ignore
            tree_pointer.append(terminal_subtree)

        for cell in cells[1:]:
            if cell.height_diff > 0:
                # Grow (height_diff) branches from the current pointer
                # Firstly, grow (height_diff - 1) empty branches
                for _ in range(cell.height_diff - 1):
                    child: Tree[NODE, LEAF] = Tree(None, [])  # type: ignore
                    tree_pointer[-1].append(child)  # type: ignore
                    tree_pointer.append(child)

                # Then create a subtree according to the cell.
                terminal_subtree = Tree.fromlist_as_unary(
                    (cell.phrase_cat, cell.lex_cat), (cell.form,)
                )
                # Lastly, adjoin it to the current position
                tree_pointer[-1].append(terminal_subtree)  # type: ignore
                tree_pointer.append(terminal_subtree)
            elif cell.height_diff == 0:
                # adjoint form to the pointer
                # (the relevant node on the last branch)
                tree_pointer[-1].append(  # type: ignore
                    Tree.fromlist_as_unary((cell.lex_cat,), (cell.form,))
                )
                # no need to move the pointer
            else:
                # adjoint form to the pointer
                # (the relevant node on the last branch)
                tree_pointer[-1].append(  # type: ignore
                    Tree.fromlist_as_unary((cell.lex_cat,), (cell.form,))
                )  # type: ignore

                # move back the pointer
                tree_pointer = tree_pointer[: cell.height_diff]

                # fill the category by recreating the tree
                latest_common_node = tree_pointer.pop()
                latest_common_node.set_label(cell.phrase_cat)

                if tree_pointer:
                    tree_pointer[-1][-1] = latest_common_node  # type: ignore
                tree_pointer.append(latest_common_node)

        return tree_pointer[0]
