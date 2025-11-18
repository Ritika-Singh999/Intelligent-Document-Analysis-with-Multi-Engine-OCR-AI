from fastapi import APIRouter

router = APIRouter()

@router.get('/reports')
async def reports_index():
    return {"reports": []}
