"""Tests for annotation module."""

from __future__ import annotations

import pytest

from sahi.logger import logger


class TestAnnotation:
    """Test cases for SAHI annotation classes."""

    def test_bounding_box(self) -> None:
        """Test BoundingBox creation and property access."""
        from sahi.annotation import BoundingBox

        bbox_minmax = [30.0, 30.0, 100.0, 150.0]
        shift_amount = [50, 40]

        bbox = BoundingBox(bbox_minmax)
        expanded_bbox = bbox.get_expanded_box(ratio=0.1)

        bbox = BoundingBox(bbox_minmax, shift_amount=shift_amount)  # type: ignore[arg-type]
        shifted_bbox = bbox.get_shifted_box()

        # compare
        assert expanded_bbox.to_xywh() == [23.0, 18.0, 84.0, 144.0]
        assert expanded_bbox.to_xyxy() == [23.0, 18.0, 107.0, 162.0]
        assert shifted_bbox.to_xyxy() == [80.0, 70.0, 150.0, 190.0]

    def test_bounding_box_immutability(self) -> None:
        """Test that BoundingBox instances are immutable."""
        import dataclasses

        from sahi.annotation import BoundingBox

        bbox_tuple = (10.0, 20.0, 30.0, 40.0)
        bbox = BoundingBox(bbox_tuple)

        # Attempt to mutate the box tuple directly
        with pytest.raises(TypeError):
            bbox.box[0] = 99.0  # type: ignore[index, call-overload]

        # Attempt to mutate the shift_amount tuple directly
        with pytest.raises(TypeError):
            bbox.shift_amount[0] = 99  # type: ignore[index]

        # Attempt to assign a new value to an attribute
        with pytest.raises(dataclasses.FrozenInstanceError):
            bbox.box = (1.0, 2.0, 3.0, 4.0)  # type: ignore[misc]

        # Attempt to assign a new value to a property
        with pytest.raises(dataclasses.FrozenInstanceError):
            bbox.minx = 123.0  # type: ignore[misc]

        # Confirm the values remain unchanged
        assert bbox.box == bbox_tuple
        assert bbox.shift_amount == (0, 0)

    def test_category(self) -> None:
        """Test Category creation and type validation."""
        from sahi.annotation import Category

        category_id = 1
        category_name = "car"
        category = Category(id=category_id, name=category_name)
        assert category.id == category_id
        assert category.name == category_name

        # id must be int
        with pytest.raises(TypeError):
            Category(id="not-an-int", name="car")  # type: ignore[arg-type]

        # name must be str
        with pytest.raises(TypeError):
            Category(id=1, name=123)  # type: ignore[arg-type]

    def test_category_immutability(self) -> None:
        """Test that Category instances are immutable."""
        import dataclasses

        from sahi.annotation import Category

        category = Category(id=5, name="person")

        # Attempt to mutate the id directly
        with pytest.raises(dataclasses.FrozenInstanceError):
            category.id = 10  # type: ignore[misc]

        # Attempt to mutate the name directly
        with pytest.raises(dataclasses.FrozenInstanceError):
            category.name = "cat"  # type: ignore[misc]

        # Confirm the values remain unchanged
        assert category.id == 5
        assert category.name == "person"

    def test_mask(self) -> None:
        """Test Mask creation from COCO segmentation format."""
        from sahi.annotation import Mask

        coco_segmentation = [[1.0, 1.0, 325.0, 125.0, 250.0, 200.0, 5.0, 200.0]]
        full_shape_height, full_shape_width = 500, 600
        full_shape = [full_shape_height, full_shape_width]

        mask = Mask(segmentation=coco_segmentation, full_shape=full_shape)

        assert mask.full_shape_height == full_shape_height
        assert mask.full_shape_width == full_shape_width
        logger.debug(f"{type(mask.bool_mask[11, 2])=} {mask.bool_mask[11, 2]=}")
        assert mask.bool_mask[11, 2]

    def test_object_annotation(self) -> None:
        """Test ObjectAnnotation creation from various formats."""
        from sahi.annotation import ObjectAnnotation

        bbox = [100, 200, 150, 230]
        coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        category_id = 2
        category_name = "car"
        shift_amount = [0, 0]
        image_height = 1080
        image_width = 1920
        full_shape = [image_height, image_width]

        object_annotation1 = ObjectAnnotation(
            bbox=bbox,  # type: ignore[arg-type]
            category_id=category_id,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

        object_annotation2 = ObjectAnnotation.from_coco_annotation_dict(
            annotation_dict={"bbox": coco_bbox, "category_id": category_id, "segmentation": []},
            category_name=category_name,
            full_shape=full_shape,
            shift_amount=shift_amount,
        )

        object_annotation3 = ObjectAnnotation.from_coco_bbox(
            bbox=coco_bbox,  # type: ignore[arg-type]
            category_id=category_id,
            category_name=category_name,
            full_shape=full_shape,
            shift_amount=shift_amount,
        )

        assert object_annotation1.bbox.minx == bbox[0]
        assert object_annotation1.bbox.miny == bbox[1]
        assert object_annotation1.bbox.maxx == bbox[2]
        assert object_annotation1.bbox.maxy == bbox[3]
        assert object_annotation1.category.id == category_id
        assert object_annotation1.category.name == category_name

        assert object_annotation2.bbox.minx == bbox[0]
        assert object_annotation2.bbox.miny == bbox[1]
        assert object_annotation2.bbox.maxx == bbox[2]
        assert object_annotation2.bbox.maxy == bbox[3]
        assert object_annotation2.category.id == category_id
        assert object_annotation2.category.name == category_name

        assert object_annotation3.bbox.minx == bbox[0]
        assert object_annotation3.bbox.miny == bbox[1]
        assert object_annotation3.bbox.maxx == bbox[2]
        assert object_annotation3.bbox.maxy == bbox[3]
        assert object_annotation3.category.id == category_id
        assert object_annotation3.category.name == category_name


class TestMaskBoolMaskCache:
    """Test that Mask.from_bool_mask preserves the original bool mask."""

    def test_from_bool_mask_stores_cache(self):
        """from_bool_mask should store _bool_mask on the Mask object."""
        from sahi.annotation import Mask
        import numpy as np

        bool_mask = np.zeros((10, 10), dtype=bool)
        bool_mask[2:5, 3:7] = True

        mask = Mask.from_bool_mask(
            bool_mask=bool_mask,
            full_shape=[10, 10],
            shift_amount=[0, 0],
        )

        # Cache must be set (not None) and return the exact original
        assert mask._bool_mask is not None
        np.testing.assert_array_equal(mask.bool_mask, bool_mask)

    def test_from_bool_mask_returns_exact_original(self):
        """bool_mask property should return the cached array, not a polygon reconstruction."""
        from sahi.annotation import Mask
        import numpy as np

        bool_mask = np.zeros((10, 10), dtype=bool)
        bool_mask[2:5, 3:7] = True

        mask = Mask.from_bool_mask(
            bool_mask=bool_mask,
            full_shape=[10, 10],
            shift_amount=[0, 0],
        )

        # Must be the same object (identity check), not just equal values
        assert mask.bool_mask is bool_mask

    def test_get_shifted_mask_preserves_bool_mask(self):
        """Shifted mask should place pixels exactly via _bool_mask cache."""
        from sahi.annotation import Mask
        import numpy as np

        bool_mask = np.zeros((10, 10), dtype=bool)
        bool_mask[0:5, 0:5] = True

        mask = Mask.from_bool_mask(
            bool_mask=bool_mask,
            full_shape=[30, 30],
            shift_amount=[10, 5],
        )

        shifted = mask.get_shifted_mask()

        # Shifted mask must also have _bool_mask set
        assert shifted._bool_mask is not None
        # Mask should be placed at (y=5, x=10) in the 30x30 canvas
        expected = np.zeros((30, 30), dtype=bool)
        expected[5:10, 10:15] = True
        np.testing.assert_array_equal(shifted.bool_mask, expected)

    def test_shifted_mask_clips_to_full_shape(self):
        """Bool mask extending beyond full_shape should be clipped."""
        from sahi.annotation import Mask
        import numpy as np

        bool_mask = np.zeros((10, 10), dtype=bool)
        bool_mask[:] = True

        mask = Mask.from_bool_mask(
            bool_mask=bool_mask,
            full_shape=[15, 15],
            shift_amount=[8, 8],
        )

        shifted = mask.get_shifted_mask()
        assert shifted._bool_mask is not None
        result = shifted.bool_mask
        assert result.shape == (15, 15)
        expected = np.zeros((15, 15), dtype=bool)
        expected[8:15, 8:15] = True
        np.testing.assert_array_equal(result, expected)

    def test_mask_without_bool_mask_cache_unchanged(self):
        """Standard Mask construction (from segmentation) should work as before."""
        from sahi.annotation import Mask
        from sahi.utils.cv import get_coco_segmentation_from_bool_mask
        import numpy as np

        bool_mask = np.zeros((10, 10), dtype=bool)
        bool_mask[2:5, 3:7] = True
        segmentation = get_coco_segmentation_from_bool_mask(bool_mask)

        # Construct via segmentation (not from_bool_mask) — no cache
        mask = Mask(
            segmentation=segmentation,
            full_shape=[10, 10],
            shift_amount=[0, 0],
        )

        assert mask._bool_mask is None
        # bool_mask property should still work via polygon reconstruction
        result = mask.bool_mask
        assert result.shape == (10, 10)

    def test_object_prediction_shifted_preserves_bool_mask(self):
        """get_shifted_object_prediction should preserve _bool_mask through shifting."""
        from sahi.prediction import ObjectPrediction
        from sahi.utils.cv import get_coco_segmentation_from_bool_mask
        import numpy as np

        # 50x50 slice mask with a 20x20 True block at origin
        bool_mask = np.zeros((50, 50), dtype=bool)
        bool_mask[0:20, 0:20] = True
        segmentation = get_coco_segmentation_from_bool_mask(bool_mask)

        pred = ObjectPrediction(
            bbox=[0, 0, 20, 20],
            category_id=0,
            score=0.9,
            segmentation=segmentation,
            category_name="crack",
            shift_amount=[20, 10],
            full_shape=[100, 100],
        )
        # Attach bool mask cache (as our wrapper does)
        pred.mask._bool_mask = bool_mask

        shifted = pred.get_shifted_object_prediction()

        # _bool_mask should be preserved on the shifted prediction
        assert shifted.mask._bool_mask is not None
        assert shifted.mask._bool_mask.shape == (100, 100)
        # 50x50 mask placed at (y=10, x=20), True block at [0:20, 0:20]
        # → canvas[10:30, 20:40] = True
        expected = np.zeros((100, 100), dtype=bool)
        expected[10:30, 20:40] = True
        np.testing.assert_array_equal(shifted.mask.bool_mask, expected)
