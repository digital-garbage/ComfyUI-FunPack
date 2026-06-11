from movie_editor.backend.nle_effects import zoom_scale_at_frame, zoompan_z_expr


def test_zoom_in_ramp_window():
    fx = {"zoom_ratio": 0.2, "zoom_start_frame": 10, "zoom_frames": 20}
    assert zoom_scale_at_frame("in", fx, 0, 100) == 1.0
    assert zoom_scale_at_frame("in", fx, 10, 100) == 1.0
    assert abs(zoom_scale_at_frame("in", fx, 20, 100) - 1.1) < 1e-6
    assert abs(zoom_scale_at_frame("in", fx, 30, 100) - 1.2) < 1e-6
    assert abs(zoom_scale_at_frame("in", fx, 50, 100) - 1.2) < 1e-6


def test_zoom_out_ramp_window():
    fx = {"zoom_ratio": 0.2, "zoom_start_frame": 0, "zoom_frames": 10}
    assert abs(zoom_scale_at_frame("out", fx, 0, 50) - 1.2) < 1e-6
    assert abs(zoom_scale_at_frame("out", fx, 5, 50) - 1.1) < 1e-6
    assert abs(zoom_scale_at_frame("out", fx, 10, 50) - 1.0) < 1e-6


def test_zoompan_expr_contains_window():
    fx = {"zoom_ratio": 0.15, "zoom_start_frame": 5, "zoom_frames": 15}
    z = zoompan_z_expr("in", fx, 80)
    assert "if(lt(on,5)" in z
    assert "if(lt(on,20)" in z
