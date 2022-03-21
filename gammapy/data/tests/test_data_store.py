# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
import astropy.units as u
from gammapy.data import DataStore
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@pytest.fixture()
def data_store():
    return DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


@requires_data()
def test_datastore_hd_hap(data_store):
    """Test HESS HAP-HD data access."""
    obs = data_store.obs(obs_id=23523)

    assert obs.events.__class__.__name__ == "EventList"
    assert obs.gti.__class__.__name__ == "GTI"
    assert obs.aeff.__class__.__name__ == "EffectiveAreaTable2D"
    assert obs.edisp.__class__.__name__ == "EnergyDispersion2D"
    assert obs.psf.__class__.__name__ == "PSF3D"


@requires_data()
def test_datastore_from_dir():
    """Test the `from_dir` method."""
    data_store_rel_path = DataStore.from_dir(
        "$GAMMAPY_DATA/hess-dl3-dr1/", "hdu-index.fits.gz", "obs-index.fits.gz"
    )

    data_store_abs_path = DataStore.from_dir(
        "$GAMMAPY_DATA/hess-dl3-dr1/",
        "$GAMMAPY_DATA/hess-dl3-dr1/hdu-index.fits.gz",
        "$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz",
    )

    assert "Data store" in data_store_rel_path.info(show=False)
    assert "Data store" in data_store_abs_path.info(show=False)


@requires_data()
def test_datastore_from_file(tmpdir):
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/hdu-index.fits.gz"
    index_hdu = fits.open(make_path(filename))["HDU_INDEX"]

    filename = "$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz"
    obs_hdu = fits.open(make_path(filename))["OBS_INDEX"]

    hdulist = fits.HDUList()
    hdulist.append(index_hdu)
    hdulist.append(obs_hdu)

    filename = tmpdir / "test-index.fits"
    hdulist.writeto(str(filename))

    data_store = DataStore.from_file(filename)

    assert data_store.obs_table["OBS_ID"][0] == 20136


@requires_data()
def test_datastore_from_events():
    # Test that `DataStore.from_events_files` works.
    # The real tests for `DataStoreMaker` are below.
    path = "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
    data_store = DataStore.from_events_files([path])
    assert len(data_store.obs_table) == 1
    assert len(data_store.hdu_table) == 6

    @requires_data()
    def test_datastore_get_observations(data_store, caplog):
        """Test loading data and IRF files via the DataStore"""
        observations = data_store.get_observations([23523, 23592])
        assert observations[0].obs_id == 23523
        observations = data_store.get_observations()
        assert len(observations) == 105

        with pytest.raises(ValueError):
            data_store.get_observations([11111, 23592])

        observations = data_store.get_observations([11111, 23523], skip_missing=True)
        assert observations[0].obs_id == 23523
        assert "WARNING" in [_.levelname for _ in caplog.records]
        assert "Skipping missing obs_id: 11111" in [_.message for _ in caplog.records]


@requires_data()
def test_broken_links_datastore(data_store):
    # Test that datastore without complete IRFs are properly loaded
    hdu_table = data_store.hdu_table
    index = np.where(hdu_table["OBS_ID"] == 23526)[0][0]
    hdu_table.remove_row(index)
    hdu_table._hdu_type_stripped = np.array([_.strip() for _ in hdu_table["HDU_TYPE"]])
    observations = data_store.get_observations(
        [23523, 23526], required_irf=["aeff", "bkg"]
    )
    assert len(observations) == 1

    with pytest.raises(ValueError):
        _ = data_store.get_observations([23523], required_irf=["xyz"])


@requires_data()
def test_datastore_copy_obs(tmp_path, data_store):
    data_store.copy_obs([23523, 23592], tmp_path, overwrite=True)

    substore = DataStore.from_dir(tmp_path)

    assert str(substore.hdu_table.base_dir) == str(tmp_path)
    assert len(substore.obs_table) == 2

    desired = data_store.obs(23523)
    actual = substore.obs(23523)

    assert str(actual.events.table) == str(desired.events.table)


@requires_data()
def test_datastore_copy_obs_subset(tmp_path, data_store):
    # Copy only certain HDU classes
    data_store.copy_obs([23523, 23592], tmp_path, hdu_class=["events"])

    substore = DataStore.from_dir(tmp_path)
    assert len(substore.hdu_table) == 2


@requires_data()
class TestDataStoreChecker:
    def setup(self):
        self.data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")

    def test_check_all(self):
        records = list(self.data_store.check())
        assert len(records) == 32


@requires_data("gammapy-data")
class TestDataStoreMaker:
    def setup(self):
        paths = [
            f"$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_{obs_id:06d}.fits"
            for obs_id in [110380, 111140, 111630, 111159]
        ]
        self.data_store = DataStore.from_events_files(paths)

        # Useful for debugging:
        # self.data_store.hdu_table.write("hdu-index.fits.gz", overwrite=True)
        # self.data_store.obs_table.write("obs-index.fits.gz", overwrite=True)

    def test_obs_table(self):
        table = self.data_store.obs_table
        assert table.__class__.__name__ == "ObservationTable"
        assert len(table) == 4
        assert len(table.colnames) == 21

        # TODO: implement https://github.com/gammapy/gammapy/issues/1218 and add tests here
        # assert table.time_start[0].iso == "spam"
        # assert table.time_start[-1].iso == "spam"

    def test_hdu_table(self):
        table = self.data_store.hdu_table
        assert table.__class__.__name__ == "HDUIndexTable"
        assert len(table) == 24
        hdu_class = ["events", "gti", "aeff_2d", "edisp_2d", "psf_3gauss", "bkg_3d"]
        assert list(self.data_store.hdu_table["HDU_CLASS"]) == 4 * hdu_class

        assert table["FILE_DIR"][2] == "$CALDB/data/cta/1dc/bcf/South_z20_50h"

    def test_observation(self, monkeypatch):
        """Check that one observation can be accessed OK"""
        obs = self.data_store.obs(110380)
        assert obs.obs_id == 110380

        assert obs.events.time[0].iso == "2021-01-21 12:00:03.045"
        assert obs.gti.time_start[0].iso == "2021-01-21 12:00:00.000"

        # Note: IRF access requires the CALDB env var
        caldb_path = Path(os.environ["GAMMAPY_DATA"]) / Path("cta-1dc/caldb")
        monkeypatch.setenv("CALDB", str(caldb_path))

        assert obs.aeff.__class__.__name__ == "EffectiveAreaTable2D"
        assert obs.bkg.__class__.__name__ == "Background3D"
        assert obs.edisp.__class__.__name__ == "EnergyDispersion2D"
        assert obs.psf.__class__.__name__ == "EnergyDependentMultiGaussPSF"


@requires_data('gammapy-data')
def test_datastore_fixed_rad_max():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/joint-crab/dl3/magic")
    observations = data_store.get_observations([5029748], required_irf=['aeff', 'edisp'])

    assert len(observations) == 1
    obs = observations[0]

    assert obs.rad_max is not None
    assert obs.rad_max.quantity.shape == (1, 1)
    assert u.allclose(obs.rad_max.quantity, np.sqrt(0.02) * u.deg)

    # test it also works with edisp (removing aeff)
    obs = data_store.get_observations([5029748], required_irf=['aeff', 'edisp'])[0]
    obs.aeff = None
    assert obs.rad_max is not None
    assert obs.rad_max.quantity.shape == (1, 1)
    assert u.allclose(obs.rad_max.quantity, 0.1414213 * u.deg)

    # removing the last irf means we have no rad_max info
    obs = data_store.get_observations([5029748], required_irf=['aeff', 'edisp'])[0]
    obs.aeff = None
    obs.edisp = None
    assert obs.rad_max is None


@requires_data()
def test_datastore_header_info_in_obs_info(data_store):
    '''Test information from the obs index header is propagated into obs_info'''
    obs = data_store.obs(obs_id=23523)

    assert "MJDREFI" in obs.obs_info
    assert "MJDREFF" in obs.obs_info
    assert "GEOLON" in obs.obs_info
    assert "GEOLAT" in obs.obs_info
    # make sure we don't add the OBS_INDEX HDUCLAS
    assert "HDUCLAS1" not in obs.obs_info

@requires_data()
def test_datastore_bad_name():
    with pytest.raises(IOError):
        DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/", "hdu-index.fits.gz", "bad")


@requires_data()
def test_datastore_from_dir_no_obs_index(caplog, tmpdir):
    """Test the `from_dir` method."""

    # Create small datastore and remove obs-index table
    DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/").copy_obs([23523, 23592], tmpdir)
    os.remove(tmpdir / "obs-index.fits.gz")

    data_store = DataStore.from_dir(tmpdir)

    obs = data_store.obs(23523)
    observations = data_store.get_observations()

    assert data_store.obs_table is None
    assert "WARNING" in [record.levelname for record in caplog.records]
    message = "Cannot find default obs-index table."
    assert message in [record.message for record in caplog.records]
    assert "No observation index table." in data_store.info(show=False)

    assert obs.obs_info["ONTIME"] == 1687.0
    assert len(observations) == 2

    data_store.copy_obs([23523], tmpdir, overwrite=True)
    data_store_copy = DataStore.from_dir(tmpdir)
    assert len(data_store_copy.obs_ids) == 1
    assert data_store_copy.obs_table == None