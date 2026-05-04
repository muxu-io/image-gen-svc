def test_package_importable():
    import image_gen_svc

    assert image_gen_svc is not None


def test_pillow_webp_available():
    from PIL import Image

    img = Image.new("RGB", (1, 1), (0, 0, 0))
    import io

    buf = io.BytesIO()
    img.save(buf, format="WEBP")  # ensures Pillow's webp plugin is built in
    assert buf.getvalue().startswith(b"RIFF")
