from quarterly.main import main


def test_main(capsys):
    """Test the main entry point."""
    main()
    captured = capsys.readouterr()
    assert "Hello from quarterly!" in captured.out
