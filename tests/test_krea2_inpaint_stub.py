import pytest

from core.backends.krea2_inpaint import Krea2MaskedEditUnavailable, prepare_krea2_masked_edit


def test_krea2_masked_edit_remains_explicitly_unavailable():
    with pytest.raises(Krea2MaskedEditUnavailable, match="trained masked"):
        prepare_krea2_masked_edit()
