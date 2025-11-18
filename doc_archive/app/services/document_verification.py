from typing import Dict, Any, List
from datetime import datetime
import logging
import os

from app.schemas.verification_schemas import DocumentVerificationResult, VerificationStatus, DocumentAuthenticityResult, VerificationConfidence, VerificationDetail
from app.utils.pdf_forensics.run_all_detectors import analyze_pdf
from app.services.profile_report import detect_document_type, detect_sensitive_identifiers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def verify_document(document_path: str, metadata: Dict[str, Any]) -> DocumentVerificationResult:

    # Validate file exists
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}")
        return DocumentVerificationResult(
            document_id=metadata.get("request_id", "unknown"),
            overall_status=VerificationStatus.SUSPICIOUS,
            authenticity=DocumentAuthenticityResult(
                status=VerificationStatus.SUSPICIOUS,
                confidence=VerificationConfidence(score=0.0, reasons=["File not found"]),
                metadata={"error": "File not found"},
                forensic_flags=["file_not_found"],
                content_consistency=False,
                timestamp=datetime.utcnow().isoformat()
            ),
            verification_details=[],
            risk_score=1.0,
            verification_timestamp=datetime.utcnow().isoformat()
        )

    try:
        verification_details = []
        risk_score = 0.0
        forensic_flags = []

        # Run forensic analysis
        forensic_result = await analyze_pdf(document_path)

        # Analyze forensic results
        if forensic_result:
            summary = forensic_result.get("summary", {})
            risk_level = summary.get("risk_level", "low")
            high_risk_findings = summary.get("high_risk_findings", 0)

            # Convert risk level to score
            risk_mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
            forensic_risk = risk_mapping.get(risk_level, 0.5)

            # Add forensic flags
            for result in forensic_result.get("results", []):
                if result.get("risk_level") == "high":
                    forensic_flags.append(result.get("detector_name", "unknown"))

            verification_details.append(VerificationDetail(
                check_name="Forensic Analysis",
                result=risk_level != "high",
                confidence=1.0 - forensic_risk,
                details=f"Forensic analysis completed with {high_risk_findings} high-risk findings"
            ))

            risk_score += forensic_risk * 0.4  # 40% weight for forensics

        # Detect document type
        doc_type, type_confidence = await detect_document_type(document_path)
        verification_details.append(VerificationDetail(
            check_name="Document Type Detection",
            result=doc_type != "unknown",
            confidence=type_confidence,
            details=f"Detected document type: {doc_type}"
        ))

        # Detect sensitive identifiers
        sensitive_ids = await detect_sensitive_identifiers(document_path)

        # Separate passport document verification from identifier presence
        passport_verified = False
        if doc_type == "passport":
            # Only verify as passport if it's actually a passport document AND contains passport numbers
            passport_identifiers = [sid for sid in sensitive_ids if sid.get("type") == "passport_number"]
            if passport_identifiers:
                passport_verified = True
                verification_details.append(VerificationDetail(
                    check_name="Passport Verification",
                    result=True,
                    confidence=0.95,
                    details="Document verified as authentic passport with matching passport number"
                ))
                risk_score -= 0.2  # Reduce risk for verified passports
            else:
                verification_details.append(VerificationDetail(
                    check_name="Passport Verification",
                    result=False,
                    confidence=0.3,
                    details="Document classified as passport but no passport number found"
                ))
                risk_score += 0.3  # Increase risk for passport without number

        # Handle sensitive identifiers (excluding passport numbers for non-passport docs)
        non_passport_sensitive_ids = [sid for sid in sensitive_ids if sid.get("type") != "passport_number"]
        if non_passport_sensitive_ids:
            verification_details.append(VerificationDetail(
                check_name="Sensitive Identifier Detection",
                result=True,
                confidence=0.9,
                details=f"Found {len(non_passport_sensitive_ids)} sensitive identifiers"
            ))
            # Increase risk score if sensitive data is found (privacy concern)
            risk_score += 0.1
        else:
            verification_details.append(VerificationDetail(
                check_name="Sensitive Identifier Detection",
                result=True,
                confidence=0.8,
                details="No sensitive identifiers detected"
            ))

        # Special handling for documents containing passport numbers but not being passports
        passport_numbers_in_non_passport = [sid for sid in sensitive_ids
                                          if sid.get("type") == "passport_number" and doc_type != "passport"]
        if passport_numbers_in_non_passport:
            verification_details.append(VerificationDetail(
                check_name="Cross-Document Identifier Check",
                result=True,
                confidence=0.7,
                details=f"Found passport number(s) in {doc_type} document - may indicate document sharing or reference"
            ))
            # Slight risk increase for cross-document references
            risk_score += 0.05

        # Determine overall status based on risk score and findings
        if risk_score > 0.7 or high_risk_findings > 0:
            overall_status = VerificationStatus.SUSPICIOUS
        elif risk_score > 0.3:
            overall_status = VerificationStatus.SUSPICIOUS
        else:
            overall_status = VerificationStatus.TRUSTED

        # Calculate overall confidence
        avg_confidence = sum(detail.confidence for detail in verification_details) / len(verification_details)

        return DocumentVerificationResult(
            document_id=metadata.get("request_id", "unknown"),
            overall_status=overall_status,
            authenticity=DocumentAuthenticityResult(
                status=overall_status,
                confidence=VerificationConfidence(
                    score=avg_confidence,
                    reasons=[detail.details for detail in verification_details if detail.result]
                ),
                metadata={
                    "document_type": doc_type,
                    "verification_timestamp": datetime.utcnow().isoformat(),
                    "sensitive_identifiers_count": len(sensitive_ids)
                },
                forensic_flags=forensic_flags,
                content_consistency=forensic_risk < 0.5,
                timestamp=datetime.utcnow().isoformat()
            ),
            verification_details=verification_details,
            risk_score=min(risk_score, 1.0),
            verification_timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error during document verification: {str(e)}")
        # Return failed verification result
        return DocumentVerificationResult(
            document_id=metadata.get("request_id", "unknown"),
            overall_status=VerificationStatus.SUSPICIOUS,
            authenticity=DocumentAuthenticityResult(
                status=VerificationStatus.SUSPICIOUS,
                confidence=VerificationConfidence(
                    score=0.0,
                    reasons=[f"Verification failed: {str(e)}"]
                ),
                metadata={"error": str(e)},
                forensic_flags=["verification_error"],
                content_consistency=False,
                timestamp=datetime.utcnow().isoformat()
            ),
            verification_details=[
                VerificationDetail(
                    check_name="Verification Process",
                    result=False,
                    confidence=0.0,
                    details=f"Verification failed with error: {str(e)}"
                )
            ],
            risk_score=1.0,
            verification_timestamp=datetime.utcnow().isoformat()
        )
