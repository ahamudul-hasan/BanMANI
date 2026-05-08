try:
    import sentencepiece
    import tiktoken
    import google.protobuf
    print('imports_ok')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('import_check_failed:', type(e).__name__, e)
