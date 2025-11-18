"""
Universal Multilingual Document Extraction Pipeline
Supports: PDF, JPG, PNG with automatic OCR, translation, classification, and grouping
Fast-optimized with multi-engine OCR fallback
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from enum import Enum

logger = logging.getLogger(__name__)

# ============ DOCUMENT TYPE MAPPINGS ============
DOCUMENT_TYPE_MAPPINGS = {
    "payslip": "payslip",
    "pay slip": "payslip",
    "salary slip": "payslip",
    "payroll": "payslip",
    "salary statement": "payslip",
    "pay statement": "payslip",
    "earnings statement": "payslip",
    "wage slip": "payslip",
    "compensation statement": "payslip",
    "bank statement": "bank_statement",
    "account statement": "bank_statement",
    "banking statement": "bank_statement",
    "financial statement": "bank_statement",
    "passport": "passport",
    "travel document": "passport",
    "international passport": "passport",
    "utility bill": "utility_bill",
    "electricity bill": "utility_bill",
    "gas bill": "utility_bill",
    "water bill": "utility_bill",
    "phone bill": "utility_bill",
    "internet bill": "utility_bill",
    "service bill": "utility_bill",
    "tax document": "tax_document",
    "tax return": "tax_document",
    "tax form": "tax_document",
    "income tax": "tax_document",
    "tax statement": "tax_document",
    "rfc": "tax_document",
    "employment contract": "employment_contract",
    "job contract": "employment_contract",
    "work contract": "employment_contract",
    "labor contract": "employment_contract",
    "id card": "id_card",
    "identification card": "id_card",
    "identity card": "id_card",
    "national id": "id_card",
    "drivers license": "id_card",
    "driver's license": "id_card",
    "medical record": "medical_record",
    "health record": "medical_record",
    "patient record": "medical_record",
    "medical report": "medical_record",
    "insurance": "insurance",
    "insurance policy": "insurance",
    "insurance document": "insurance",
    "coverage document": "insurance",
    "visa": "visa",
    "entry visa": "visa",
    "travel visa": "visa",
    "immigration visa": "visa",
    "invoice": "invoice",
    "receipt": "receipt",
    "certificate": "certificate",
    "resume": "resume",
    "cv": "resume",
    "contract": "contract",
    "agreement": "contract",
    "unknown": "other_document"
}

# ============ LANGUAGE DETECTION & TRANSLATION ============
class LanguageDetector:
    """Detect language and translate OCR output to English."""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language using patterns and langdetect."""
        if not text:
            return "unknown"
        
        try:
            # Fast pattern-based detection
            spanish_indicators = sum(1 for word in ['el', 'la', 'de', 'que', 'en', 'español', 'año', 'nombre', 'número']
                                    if f" {word} " in f" {text.lower()} ")
            english_indicators = sum(1 for word in ['the', 'and', 'of', 'to', 'in', 'english', 'year', 'name', 'number']
                                    if f" {word} " in f" {text.lower()} ")
            
            if spanish_indicators > english_indicators:
                return "es"
            elif english_indicators > spanish_indicators:
                return "en"
            
            # Fallback to langdetect if available
            try:
                from langdetect import detect
                return detect(text[:500])
            except:
                pass
            
            return "en"  # Default to English
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return "en"
    
    @staticmethod
    async def translate_to_english(text: str, source_lang: str = None) -> Tuple[str, str]:
        """Translate text to English if needed."""
        if not text:
            return text, "unknown"
        
        if source_lang is None:
            source_lang = LanguageDetector.detect_language(text)
        
        if source_lang == "en":
            return text, "en"
        
        try:
            from google.cloud import translate_v2
            from app.core.config import settings
            
            translator = translate_v2.Client()
            result = translator.translate_text(text, source_language=source_lang, target_language='en')
            translated_text = result['translatedText']
            
            logger.info(f"Translated from {source_lang} to en")
            return translated_text, source_lang
            
        except Exception as e:
            logger.warning(f"Translation failed ({source_lang}->en): {e}, using original text")
            return text, source_lang

# ============ FAST OCR ORCHESTRATOR ============
class FastOCROrchestrator:
    """Multi-engine OCR with fastest-first strategy."""
    
    @staticmethod
    async def extract_text_fast(image_data: bytes, use_all_engines: bool = False) -> Tuple[str, float, str]:
        """
        Extract text from image using fastest available OCR engine.
        Returns: (text, confidence, engine_name)
        """
        from PIL import Image
        import io
        
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Image load failed: {e}")
            return "", 0.0, "error"
        
        engines_tried = []
        
        # Try PaddleOCR first (fastest + accurate)
        try:
            from app.core.ocr_engines import extract_with_paddle
            text, conf = extract_with_paddle(image)
            engines_tried.append(("paddle", conf))
            if text and conf >= 0.6:
                logger.info(f"OCR: PaddleOCR ({conf:.2f})")
                return text, conf, "paddle"
        except Exception as e:
            logger.debug(f"PaddleOCR failed: {e}")
        
        # Try ONNX (lightweight alternative)
        try:
            from app.core.ocr_engines import extract_with_onnx
            text, conf = extract_with_onnx(image)
            engines_tried.append(("onnx", conf))
            if text and conf >= 0.6:
                logger.info(f"OCR: ONNX ({conf:.2f})")
                return text, conf, "onnx"
        except Exception as e:
            logger.debug(f"ONNX failed: {e}")
        
        # Fallback to Tesseract
        try:
            from app.core.ocr_engines import extract_with_tesseract
            text, conf = extract_with_tesseract(image)
            engines_tried.append(("tesseract", conf))
            if text:
                logger.info(f"OCR: Tesseract ({conf:.2f})")
                return text, conf, "tesseract"
        except Exception as e:
            logger.debug(f"Tesseract failed: {e}")
        
        logger.error(f"All OCR engines failed. Tried: {engines_tried}")
        return "", 0.0, "none"

# ============ FAST DOCUMENT CLASSIFIER ============
class FastDocumentClassifier:
    """Semantic + keyword-based document type detection."""
    
    @staticmethod
    def classify_document(text: str) -> Tuple[str, float]:
        """
        Classify document type using keyword scoring + semantic patterns.
        Returns: (document_type, confidence)
        """
        if not text:
            return "other_document", 0.0
        
        text_lower = text.lower()
        
        # Keyword patterns for each document type
        type_patterns = {
            "payslip": {
                "keywords": ["salario", "sueldo", "salary", "gross pay", "net pay", "employer", 
                           "employee", "nómina", "salary statement", "earnings", "payroll"],
                "weight": 1.0
            },
            "tax_document": {
                "keywords": ["rfc", "impuesto", "tax", "fiscal", "declaración", "income tax",
                           "tax return", "tax form", "deduction", "tributario", "contribuyente"],
                "weight": 1.0
            },
            "invoice": {
                "keywords": ["factura", "invoice", "invoice number", "amount", "total", "fecha",
                           "fecha de emisión", "razón social", "rfc", "iva", "subtotal"],
                "weight": 1.0
            },
            "receipt": {
                "keywords": ["recibo", "receipt", "amount", "total", "date", "fecha", "pagado",
                           "concepto", "descripción", "price"],
                "weight": 0.95
            },
            "bank_statement": {
                "keywords": ["banco", "bank", "balance", "cuenta", "statement", "transaction",
                           "depósito", "extracción", "transferencia", "movimiento", "estado de cuenta"],
                "weight": 1.0
            },
            "passport": {
                "keywords": ["passport", "pasaporte", "mrz", "nationality", "date of birth",
                           "fecha de nacimiento", "número de pasaporte", "expedición", "vigencia"],
                "weight": 1.0
            },
            "employment_contract": {
                "keywords": ["contrato", "contract", "empleado", "employee", "cargo", "position",
                           "empresa", "company", "sueldo", "funciones", "jornada"],
                "weight": 1.0
            },
            "id_card": {
                "keywords": ["identification", "id card", "carnet", "cédula", "documento",
                           "nacional de identidad", "number", "fecha de nacimiento", "sexo"],
                "weight": 0.9
            },
            "utility_bill": {
                "keywords": ["utility", "bill", "electric", "gas", "water", "phone", "internet",
                           "factura de servicios", "recibo de servicio", "kilovatio", "m³"],
                "weight": 0.9
            },
            "insurance": {
                "keywords": ["insurance", "policy", "póliza", "cobertura", "prima", "vigencia",
                           "asegurado", "beneficiario", "condiciones"],
                "weight": 0.9
            },
            "visa": {
                "keywords": ["visa", "entry visa", "travel visa", "immigration", "visado",
                           "tipo de visa", "fecha de emisión", "país"],
                "weight": 0.95
            },
            "medical_record": {
                "keywords": ["medical", "health", "patient", "médico", "paciente", "diagnosis",
                           "diagnóstico", "tratamiento", "médico", "medicamento"],
                "weight": 0.9
            },
            "certificate": {
                "keywords": ["certificate", "certificado", "certified", "certification",
                           "acta", "constancia", "sello", "firma autorizada"],
                "weight": 0.85
            },
            "resume": {
                "keywords": ["resume", "cv", "curriculum", "experience", "education",
                           "skills", "experiencia", "educación", "habilidades"],
                "weight": 0.85
            },
        }
        
        # Score each document type
        scores = {}
        for doc_type, pattern_data in type_patterns.items():
            keywords = pattern_data["keywords"]
            weight = pattern_data["weight"]
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text_lower)
            match_ratio = matches / len(keywords) if keywords else 0
            
            # Calculate confidence
            confidence = (match_ratio ** 0.5) * weight  # Square root for smoothing
            scores[doc_type] = confidence
        
        # Find best match
        best_type = max(scores, key=scores.get) if scores else "other_document"
        best_confidence = scores.get(best_type, 0.0)
        
        # Normalize confidence to 0-1 range
        final_confidence = min(0.99, best_confidence) if best_confidence > 0 else 0.3
        
        logger.debug(f"Classification: {best_type} ({final_confidence:.2f})")
        return best_type, final_confidence

# ============ FAST FIELD EXTRACTOR ============
class FastFieldExtractor:
    """Extract key fields using regex + semantic patterns."""
    
    @staticmethod
    def extract_owner(text: str) -> Tuple[str, float]:
        """Extract owner/person name using regex + spaCy."""
        if not text:
            return "Unknown", 0.0
        
        try:
            import spacy
            import re
            
            # Try spaCy NER first
            try:
                nlp = spacy.load("es_core_news_sm")
            except:
                try:
                    nlp = spacy.load("en_core_web_sm")
                except:
                    nlp = spacy.blank("es")
            
            doc = nlp(text[:3000])  # Limit text for speed
            
            # Extract person entities
            persons = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "PER"]]
            if persons:
                return persons[0], 0.85
            
            # Regex fallback: look for name patterns
            patterns = [
                r'(?:Sr\.|Sra\.|Mr\.|Ms\.|Dr\.|Ing\.)\s+([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)*)',
                r'nombre[:\s]+([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)*)',
                r'name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'^([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){1,3})$',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    return match.group(1).strip(), 0.75
            
            return "Unknown", 0.0
            
        except Exception as e:
            logger.debug(f"Owner extraction failed: {e}")
            return "Unknown", 0.0
    
    @staticmethod
    def extract_passport_data(text: str) -> Dict[str, Any]:
        """
        Extract passport information including MRZ, number, dates.
        Checks EVERY document for passport presence, even non-passports.
        """
        result = {
            "passportDetected": False,
            "passportNumber": None,
            "mrz": None,
            "dob": None,
            "expiryDate": None,
            "nationality": None,
            "name": None
        }
        
        if not text:
            return result
        
        try:
            # Machine Readable Zone (MRZ) pattern
            mrz_pattern = r'[A-Z]{2}[A-Z0-9<]{39}[0-9]{9}'
            mrz_match = re.search(mrz_pattern, text)
            if mrz_match:
                result["mrz"] = mrz_match.group()
                result["passportDetected"] = True
            
            # Passport number patterns (various formats)
            passport_patterns = [
                r'(?:pasaporte|passport)[\s:]*([A-Z0-9]{6,15})',
                r'(?:número|number|no\.|#)[\s:]*([A-Z]{1,2}[0-9]{6,10})',
                r'([A-Z]{2}[0-9]{7})',  # Standard format: 2 letters + 7 digits
                r'([A-Z0-9]{8,15})',     # Generic alphanumeric
            ]
            
            for pattern in passport_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and len(match.group(1)) >= 6:
                    result["passportNumber"] = match.group(1).upper()
                    result["passportDetected"] = True
                    break
            
            # Date of birth patterns
            dob_patterns = [
                r'(?:fecha de nacimiento|date of birth|dob)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
            ]
            
            for pattern in dob_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["dob"] = match.group(1)
                    break
            
            # Expiry date patterns
            expiry_patterns = [
                r'(?:vigencia|expiry|valid until|expires)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?:hasta|valid)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            ]
            
            for pattern in expiry_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["expiryDate"] = match.group(1)
                    break
            
            # Nationality patterns
            nationality_pattern = r'(?:nationality|nacionalidad)[\s:]*([A-Za-z\s]+?)(?:\n|,|$)'
            nationality_match = re.search(nationality_pattern, text, re.IGNORECASE)
            if nationality_match:
                result["nationality"] = nationality_match.group(1).strip()
            
        except Exception as e:
            logger.debug(f"Passport extraction failed: {e}")
        
        return result
    
    @staticmethod
    def extract_key_fields(text: str, doc_type: str) -> Dict[str, Any]:
        """Extract type-specific key fields."""
        fields = {
            "documentType": doc_type,
            "extractedFields": {},
            "confidence": 0.75
        }
        
        try:
            text_lower = text.lower()
            
            if doc_type == "payslip":
                # Extract employer, net pay, gross pay, etc.
                patterns = {
                    "employerName": r'(?:employer|empresa|razón social)[\s:]*([^\n]+)',
                    "netPay": r'(?:net pay|sueldo neto|net income)[\s:]*\$?([0-9,\.]+)',
                    "grossPay": r'(?:gross pay|sueldo bruto|gross income)[\s:]*\$?([0-9,\.]+)',
                    "employeeId": r'(?:employee id|id empleado|número|employee)[\s:]*([A-Z0-9]+)',
                }
            
            elif doc_type == "tax_document":
                patterns = {
                    "rfc": r'(?:rfc|tax id)[\s:]*([A-ZÑ]{3,4}\d{6}[A-Z0-9]{3})',
                    "fiscalYear": r'(?:fiscal year|año fiscal|ejercicio)[\s:]*(\d{4})',
                    "taxableIncome": r'(?:taxable|ingresos gravables)[\s:]*\$?([0-9,\.]+)',
                    "totalTax": r'(?:total tax|impuesto total)[\s:]*\$?([0-9,\.]+)',
                }
            
            elif doc_type == "invoice":
                patterns = {
                    "invoiceId": r'(?:invoice|factura)[\s#:]*([0-9]+)',
                    "invoiceDate": r'(?:date|fecha)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    "totalAmount": r'(?:total)[\s:]*\$?([0-9,\.]+)',
                    "taxAmount": r'(?:iva|tax)[\s:]*\$?([0-9,\.]+)',
                }
            
            elif doc_type == "receipt":
                patterns = {
                    "receiptNumber": r'(?:receipt|recibo)[\s#:]*([0-9]+)',
                    "receiptDate": r'(?:date|fecha)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    "totalAmount": r'(?:total)[\s:]*\$?([0-9,\.]+)',
                }
            
            elif doc_type == "bank_statement":
                patterns = {
                    "accountNumber": r'(?:account|cuenta)[\s#:]*([0-9]+)',
                    "balance": r'(?:balance|saldo)[\s:]*\$?([0-9,\.]+)',
                    "statementDate": r'(?:statement date|fecha)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                }
            
            elif doc_type == "passport":
                patterns = {
                    "passportId": r'(?:passport|pasaporte)[\s#:]*([A-Z0-9]{6,15})',
                    "issueDate": r'(?:issue date|fecha de emisión)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    "expiryDate": r'(?:expiry|vigencia)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                }
            
            else:
                patterns = {}
            
            # Extract using patterns
            for field_name, pattern in patterns.items():
                try:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        fields["extractedFields"][field_name] = match.group(1).strip()
                except:
                    pass
            
        except Exception as e:
            logger.debug(f"Field extraction failed: {e}")
        
        return fields

# ============ UNIVERSAL EXTRACTION ORCHESTRATOR ============
class UniversalDocumentExtractor:
    """
    Main orchestrator for universal multilingual document extraction.
    Handles: Download → OCR → Translate → Classify → Extract → Group → Format
    """
    
    def __init__(self):
        self.ocr = FastOCROrchestrator()
        self.classifier = FastDocumentClassifier()
        self.extractor = FastFieldExtractor()
        self.lang_detector = LanguageDetector()
    
    async def process_document_url(self, url: str, url_index: int = 0) -> Dict[str, Any]:
        """
        Process single document from URL.
        Returns: Document extraction result
        """
        try:
            # Download
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        return {
                            "status": "error",
                            "url": url,
                            "error": f"HTTP {resp.status}",
                            "fileName": f"doc_{url_index}",
                            "documentType": "unknown",
                            "confidence": 0.0
                        }

                    # Validate content type
                    content_type = resp.headers.get('content-type', '').lower()
                    if not (content_type.startswith('application/pdf') or
                           content_type.startswith('image/') or
                           'pdf' in content_type or
                           any(img_type in content_type for img_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'])):
                        logger.warning(f"Invalid content type for {url}: {content_type}")
                        return {
                            "status": "error",
                            "url": url,
                            "error": f"Invalid content type: {content_type}",
                            "fileName": f"doc_{url_index}",
                            "documentType": "unknown",
                            "confidence": 0.0
                        }

                    image_data = await resp.read()
            
            # Convert PDF to images if needed
            from PIL import Image
            import io
            
            try:
                # Try direct image
                Image.open(io.BytesIO(image_data))
                images_data = [image_data]
            except:
                # Try PDF
                try:
                    from pdf2image import convert_from_bytes
                    images = convert_from_bytes(image_data, dpi=200)
                    images_data = []
                    for img in images:
                        img_byte = io.BytesIO()
                        img.save(img_byte, format='PNG')
                        images_data.append(img_byte.getvalue())
                except:
                    logger.error(f"Cannot process {url}")
                    return {
                        "status": "error",
                        "url": url,
                        "error": "Cannot process file format",
                        "fileName": f"doc_{url_index}",
                        "documentType": "unknown",
                        "confidence": 0.0
                    }
            
            # OCR all pages
            ocr_texts = []
            ocr_confidences = []
            ocr_engines = []
            
            for img_data in images_data:
                text, conf, engine = await self.ocr.extract_text_fast(img_data)
                if text:
                    ocr_texts.append(text)
                    ocr_confidences.append(conf)
                    ocr_engines.append(engine)
            
            if not ocr_texts:
                return {
                    "status": "error",
                    "url": url,
                    "error": "OCR failed",
                    "fileName": f"doc_{url_index}",
                    "documentType": "unknown",
                    "confidence": 0.0
                }
            
            combined_text = "\n".join(ocr_texts)
            avg_ocr_conf = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0
            
            # Detect language
            detected_lang = self.lang_detector.detect_language(combined_text)
            
            # Translate to English
            translated_text, original_lang = await self.lang_detector.translate_to_english(combined_text, detected_lang)
            
            # Classify document type
            doc_type, type_conf = self.classifier.classify_document(translated_text)
            
            # Extract owner
            owner, owner_conf = self.extractor.extract_owner(translated_text)
            
            # Extract passport data (MANDATORY for every document)
            passport_data = self.extractor.extract_passport_data(translated_text)
            
            # Extract key fields
            key_fields = self.extractor.extract_key_fields(translated_text, doc_type)
            
            # Compile result
            result = {
                "status": "success",
                "url": url,
                "fileName": f"doc_{url_index}_{doc_type}",
                "documentType": doc_type,
                "documentTypeConfidence": round(type_conf, 2),
                "ownerName": owner,
                "ownerConfidence": round(owner_conf, 2),
                "ocrConfidence": round(avg_ocr_conf, 2),
                "ocrEngine": ocr_engines[0] if ocr_engines else "unknown",
                "originalLanguage": original_lang,
                "detectedLanguage": detected_lang,
                "passportDetected": passport_data["passportDetected"],
                "passportNumber": passport_data["passportNumber"],
                "extractedFields": key_fields.get("extractedFields", {}),
                "confidence": round((type_conf + owner_conf) / 2, 2),
                "pageCount": len(images_data),
                "textLength": len(combined_text)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed ({url}): {e}")
            return {
                "status": "error",
                "url": url,
                "error": str(e),
                "fileName": f"doc_{url_index}",
                "documentType": "unknown",
                "confidence": 0.0
            }
    
    async def process_batch(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process batch of document URLs and return formatted response.
        """
        start_time = datetime.utcnow()
        
        # Process all documents in parallel
        tasks = [self.process_document_url(url, i) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Group documents by type
        grouped = {}
        successful_docs = []
        failed_docs = []
        
        for result in results:
            if result["status"] == "success":
                successful_docs.append(result)
                doc_type = result["documentType"]
                if doc_type not in grouped:
                    grouped[doc_type] = []
                grouped[doc_type].append(result)
            else:
                failed_docs.append(result)
        
        # Extract summary data
        owner_names = [doc["ownerName"] for doc in successful_docs if doc["ownerName"] != "Unknown"]
        dominant_doc_type = max(grouped.keys(), key=lambda k: len(grouped[k])) if grouped else "unknown"
        languages = list(set([doc["detectedLanguage"] for doc in successful_docs]))

        has_passport = any(doc["passportDetected"] for doc in successful_docs)
        passport_numbers = [doc["passportNumber"] for doc in successful_docs if doc["passportNumber"]]

        confidences = [doc["confidence"] for doc in successful_docs]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        processing_time = (datetime.utcnow() - start_time).total_seconds()
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(urls)}_docs"

        # Enhanced failure reporting
        failure_reasons = {}
        for doc in failed_docs:
            error = doc.get("error", "Unknown error")
            if error not in failure_reasons:
                failure_reasons[error] = 0
            failure_reasons[error] += 1

        logger.info(f"Batch processing failures: {failure_reasons}")
        
        # Build final response (EXACT JSON SHAPE REQUIRED)
        response = {
            "status": "success" if len(successful_docs) == len(urls) else ("partial" if successful_docs else "failed"),
            "batchId": batch_id,
            "summary": {
                "ownerName": owner_names[0] if owner_names else "Unknown",
                "documentCount": len(urls),
                "verifiedDocuments": len(successful_docs),
                "documentsWithSensitiveData": sum(1 for doc in successful_docs if doc["passportDetected"]),
                "documentsWithFormatErrors": len(failed_docs),
                "averageConfidence": round(avg_confidence, 2),
                "dominantDocumentType": dominant_doc_type,
                "languagesDetected": languages,
                "fiscalResidency": "Unknown",  # Would need additional logic
                "employmentType": "Unknown",   # Would need additional logic
                "hasPassport": has_passport,
                "passportNumber": passport_numbers[0] if passport_numbers else None,
                "passportVerified": has_passport,
                "notes": f"Processed {len(successful_docs)}/{len(urls)} documents successfully"
            },
            "groupedDocuments": {
                doc_type: documents for doc_type, documents in grouped.items()
            },
            "keyFactors": {
                "ownerId": (owner_names[0] or "Unknown").replace(" ", "_").upper(),
                "totalDocumentsAnalyzed": len(urls),
                "highConfidenceDocs": sum(1 for doc in successful_docs if doc["confidence"] > 0.8),
                "taxId": next((doc["extractedFields"].get("rfc") for doc in successful_docs 
                             if doc["documentType"] == "tax_document"), "Unknown"),
                "passportNumber": passport_numbers[0] if passport_numbers else None,
                "primaryLanguage": languages[0] if languages else "unknown",
                "businessCategory": "Unknown",
                "financialPeriodCoverage": "Unknown",
                "incomeStability": "Unknown",
                "riskSummary": "Low risk" if avg_confidence > 0.8 else ("Medium risk" if avg_confidence > 0.6 else "High risk")
            },
            "processingSummary": {
                "systemVersion": "2.5.0-universal",
                "pipelinesUsed": ["ocr", "translate", "classify", "extract", "group"],
                "modelsUsed": ["paddle_ocr", "onnx", "spacy_ner", "regex_patterns"],
                "processingTimeSeconds": round(processing_time, 2),
                "detectedFormatWarnings": [doc.get("error") for doc in failed_docs] if failed_docs else [],
                "qualityAssurance": "passed" if len(successful_docs) == len(urls) else "review_needed",
                "verificationSummary": {
                    "documentVerified": len(successful_docs) > 0,
                    "totalDocuments": len(urls),
                    "verifiedCount": len(successful_docs)
                }
            }
        }
        
        logger.info(f"Batch processing completed: {len(successful_docs)}/{len(urls)} docs in {processing_time:.2f}s")
        return response

# ============ CONVENIENCE FUNCTION ============
async def extract_documents_universal(urls: List[str]) -> Dict[str, Any]:
    """
    Universal document extraction for any URLs (PDF/JPG/PNG).
    
    Args:
        urls: List of document URLs
    
    Returns:
        Formatted extraction result with exact JSON shape specified
    """
    extractor = UniversalDocumentExtractor()
    return await extractor.process_batch(urls)
