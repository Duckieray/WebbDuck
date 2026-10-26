"""The lightbox fetches the next gallery page while you are still browsing.

Regression: the lightbox is opened with whatever tiles the gallery has rendered, and
PhotoSwipe wraps from the last slide back to the first (``loop`` defaults to true). So a
40-image first page looked complete, and pressing next at image 40 teleported the user
to 1/40 instead of continuing into the next batch.

The paging methods are sliced out of the real module and executed in node so these
assertions cover shipped code rather than a reimplementation of it.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIGHTBOX = (ROOT / "ui" / "modules" / "LightboxManager.js").read_text(encoding="utf-8")
GALLERY = (ROOT / "ui" / "modules" / "GalleryManager.js").read_text(encoding="utf-8")


def _methods(*names: str) -> str:
    """Slice the named class methods out of LightboxManager.js, verbatim."""
    chunks = []
    for name in names:
        found = re.search(rf"\n    (?:async )?{re.escape(name)}\(", LIGHTBOX)
        assert found, f"{name} is missing from LightboxManager.js"
        start = found.start()
        depth = 0
        seen_brace = False
        for index in range(LIGHTBOX.index("{", start), len(LIGHTBOX)):
            char = LIGHTBOX[index]
            if char == "{":
                depth += 1
                seen_brace = True
            elif char == "}":
                depth -= 1
                if seen_brace and depth == 0:
                    chunks.append(LIGHTBOX[start:index + 1])
                    break
        else:  # pragma: no cover - unbalanced braces in the module
            raise AssertionError(f"unbalanced braces in {name}")
    return "\n\n".join(chunks)


def test_lightbox_opens_without_wrapping_before_the_collection_ends():
    """Wrapping must be off at open, or the next-page fetch can never be reached."""
    assert "loop: false," in LIGHTBOX, (
        "the lightbox must not wrap by default; it would jump to slide 1 mid-browse"
    )
    # The flag is re-enabled later, once there is genuinely nothing left to fetch.
    assert "pswp.options.loop = true" in LIGHTBOX


def test_paging_is_triggered_on_navigation_and_at_open():
    """Prefetch on open (opened near the end) and on every slide change."""
    assert LIGHTBOX.count("this._maybeExtendPages()") >= 2
    assert "this._maybeExtendPages();" in LIGHTBOX


def test_paging_only_runs_for_a_paged_gallery_feed():
    """A filtered/searched gallery shows one complete set, so it must not be extended."""
    paging = _methods("_pagingEligible")
    assert "gallery.activeFilter !== 'all'" in paging
    assert "gallery-search" in paging
    # Both the grid and the eligibility gate are captured from the real open() path.
    assert "this.galleryGrid = openedFromTiles" in LIGHTBOX
    assert "this.pagingEnabled = Boolean(openedFromTiles)" in LIGHTBOX
    # And the gate it relies on exists on the gallery.
    assert "shouldAutoLoadMore()" in GALLERY


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_navigating_to_the_end_loads_the_next_batch_instead_of_wrapping():
    """Execute the shipped paging methods against a mock gallery and PhotoSwipe."""
    script = f"""
let searchTerm = '';
const byId = (id) => (id === 'gallery-search' ? {{ value: searchTerm }} : null);

const makeTile = (src) => ({{
  src: '/thumbs/' + src, alt: src.split('/').pop(), naturalWidth: 896, naturalHeight: 1344,
  closest: (sel) => sel === '.image-item'
    ? {{ dataset: {{ src, width: '896', height: '1344', favorite: '0' }} }}
    : null,
}});

const Lightbox = eval(`(class Lightbox {{
  constructor(state) {{
    Object.assign(this, state);
    this.pageAhead = 3;
    this.isLoadingNextPage = false;
  }}
  extractSessionMeta() {{ return {{ prompt: 'p' }}; }}
{_methods("_slideFromTile", "_pagingEligible", "_maybeExtendPages", "_appendFreshTiles")}
}})`);

const results = [];
const check = (name, ok, detail) => results.push({{ name, ok: Boolean(ok), detail: detail ?? '' }});
const setup = (count, opts = {{}}) => {{
  const dataSource = Array.from({{ length: count }}, (_, i) => ({{ src: `outputs/run/img_${{i}}.png` }}));
  const gallery = {{
    activeFilter: opts.filter || 'all',
    hasMore: opts.hasMore !== false,
    isLoadingPage: Boolean(opts.isLoadingPage),
    loads: 0,
    async loadMoreAuto() {{ this.loads += 1; this.hasMore = false; }},
  }};
  const mgr = new Lightbox({{
    pagingEnabled: true,
    galleryGrid: {{ querySelectorAll: () => (opts.fresh || []).map(makeTile) }},
  }});
  mgr.currentPswpInstance = {{ pswp: {{ currIndex: 0, options: {{ dataSource, loop: false }} }} }};
  global.window = {{ galleryManager: gallery }};
  return {{ mgr, gallery, dataSource }};
}};

(async () => {{
  const fresh = Array.from({{ length: 5 }}, (_, i) => `outputs/run/new_${{i}}.png`);

  let t = setup(40, {{ fresh }});
  t.mgr.currentPswpInstance.pswp.currIndex = 30;
  await t.mgr._maybeExtendPages();
  check('no fetch in the middle of the list', t.gallery.loads === 0, `loads=${{t.gallery.loads}}`);

  t.mgr.currentPswpInstance.pswp.currIndex = 37;
  await t.mgr._maybeExtendPages();
  check('fetches the next page near the end', t.gallery.loads === 1, `loads=${{t.gallery.loads}}`);
  check('extends the lightbox with the new tiles', t.dataSource.length === 45, `len=${{t.dataSource.length}}`);
  check('appended slides keep their real size', t.dataSource[40].width === 896 && t.dataSource[40].height === 1344);
  check('does not wrap while more pages remain', t.mgr.currentPswpInstance.pswp.options.loop === false);

  await t.mgr._maybeExtendPages();
  check('asking again does not duplicate', t.dataSource.length === 45, `len=${{t.dataSource.length}}`);

  t = setup(10, {{ fresh: ['outputs/run/img_3.png', 'outputs/run/brand_new.png'] }});
  t.mgr.currentPswpInstance.pswp.currIndex = 9;
  await t.mgr._maybeExtendPages();
  check('skips tiles already open',
    t.dataSource.length === 11 && t.dataSource.filter(d => d.src === 'outputs/run/img_3.png').length === 1,
    `len=${{t.dataSource.length}}`);

  t = setup(10, {{ hasMore: false }});
  t.mgr.currentPswpInstance.pswp.currIndex = 9;
  await t.mgr._maybeExtendPages();
  check('restores wrapping at the true end', t.mgr.currentPswpInstance.pswp.options.loop === true);
  check('does not fetch when nothing is left', t.gallery.loads === 0);

  t = setup(10, {{ filter: 'favorites' }});
  t.mgr.currentPswpInstance.pswp.currIndex = 9;
  await t.mgr._maybeExtendPages();
  check('filtered gallery is not paged', t.gallery.loads === 0, `loads=${{t.gallery.loads}}`);
  check('filtered gallery restores wrapping', t.mgr.currentPswpInstance.pswp.options.loop === true);

  searchTerm = 'cat';
  t = setup(10);
  t.mgr.currentPswpInstance.pswp.currIndex = 9;
  await t.mgr._maybeExtendPages();
  check('searched gallery is not paged', t.gallery.loads === 0, `loads=${{t.gallery.loads}}`);
  check('searched gallery restores wrapping', t.mgr.currentPswpInstance.pswp.options.loop === true);
  searchTerm = '';

  t = setup(10);
  t.mgr.pagingEnabled = false;
  t.mgr.currentPswpInstance.pswp.currIndex = 9;
  await t.mgr._maybeExtendPages();
  check('a synthetic lightbox is never paged', t.gallery.loads === 0 && t.dataSource.length === 10);

  t = setup(40, {{ isLoadingPage: true }});
  t.mgr.currentPswpInstance.pswp.currIndex = 39;
  await t.mgr._maybeExtendPages();
  check('defers to an in-flight scroll load', t.gallery.loads === 0 && t.dataSource.length === 40);

  console.log(JSON.stringify(results));
}})();
"""
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout)

    assert results, "the paging harness produced no results"
    failures = [r for r in results if not r["ok"]]
    assert not failures, "\n".join(f"  {r['name']} {r['detail']}" for r in failures)
    assert len(results) >= 12, f"only {len(results)} behavioural checks ran"
