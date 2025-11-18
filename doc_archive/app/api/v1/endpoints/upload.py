from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import json
from werkzeug.utils import secure_filename

router = APIRouter()

@router.post('/upload')
async def upload_file(
    file: UploadFile = File(...),
    documentType: str = Form(...),
    metadata: str = Form(...)
):
    # save uploaded file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(os.getcwd(), 'documents', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)
    # write to disk
    with open(filepath, 'wb') as f:
        f.write(await file.read())
    # parse metadata
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    # process with archive
    try:
        archive = DocumentArchive("documents")
        doc_id = await archive.process_documents([filepath])
        return {
            "documentId": doc_id,
            "status": "uploaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
