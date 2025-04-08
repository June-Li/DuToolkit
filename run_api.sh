python WORKFLOW/OTHER/OCR/v0/OCRApiDeploy.py
# 调用：
# curl -X POST "http://127.0.0.1:6000/upload_ocr_result" -F "file=@pytest_scripts/data/pdf_fake_text_table.pdf" -F "json={\"force_cv\": \"yes\", \"table_enhanced\": true, \"figure_rec_by_vl\": true}" -o out.json
# curl -X POST "http://127.0.0.1:6000/upload_ocr_result" \
# -F "file=@pytest_scripts/data/pdf_fake_text_table.pdf" \
# -F "json={
#     \"force_cv\": \"yes\", 
#     \"resolution\": \"high\", 
#     \"de_seal_enable\": true, 
#     \"fix_text_enable\": true, 
#     \"char_box_enable\": true, 
#     \"tilt_correction_enable\": true, 
#     \"border_table_alg\": \"sdbordertable_cv_model\", 
#     \"borderless_table_alg\": \"sdborderlesstable_lm_lite_model\", 
#     \"layout_alg\": \"sdlayout\", 
#     \"formula_enable\": true, 
#     \"figure_rec_by_vl_enbale\": false, 
#     \"figure_upload_enable\": false, 
#     \"chapter_alg\": \"re\", 
#     \"reading_order_enable\": true
# }" \
# -o out.json