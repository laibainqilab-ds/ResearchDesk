from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.deps import get_current_user, get_document_service
from app.api.schemas import DocumentOut
from app.db.models import User
from app.ingestion.parsers import UnsupportedFileTypeError
from app.ingestion.pipeline import DocumentParsingError, DuplicateDocumentError, EmptyDocumentError
from app.services.document_service import DocumentAccessDeniedError, DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    file_bytes = await file.read()

    try:
        result = document_service.ingest(owner_id=user.id, file_bytes=file_bytes, filename=file.filename)
    except UnsupportedFileTypeError as error:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error))
    except DuplicateDocumentError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    except EmptyDocumentError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))
    except DocumentParsingError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))

    return DocumentOut(**result)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> list[DocumentOut]:
    return [DocumentOut(**document) for document in document_service.list_for_owner(user.id)]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service),
) -> None:
    try:
        document_service.delete(owner_id=user.id, document_id=document_id)
    except DocumentAccessDeniedError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
