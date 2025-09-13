"use client";
import { useState, useEffect } from "react";

interface AnalysisResult {
  type: string;
  analysis: string;
  loading: boolean;
  error?: string;
}

interface SynthesisResult {
  synthesis: string;
}

interface DischargeOrder {
  patientName: string;
  medicalRecordNumber: string;
  dateOfBirth: string;
  dischargeDate: string;
  admissionDate: string;
  primaryDiagnosis: string;
  secondaryDiagnoses: string[];
  medications: Array<{
    name: string;
    dosage: string;
    frequency: string;
    duration: string;
    instructions: string;
  }>;
  followUpInstructions: string[];
  activityRestrictions: string;
  dietInstructions: string;
  warningSignsToReturn: string[];
  dischargingPhysician: string;
  dischargeDisposition: string;
  vitalSignsAtDischarge: {
    bloodPressure: string;
    heartRate: string;
    temperature: string;
    oxygenSaturation: string;
    respiratoryRate: string;
  };
  functionalStatus: string;
  dischargeInstructions: {
    woundCare: string | null;
    equipmentNeeded: string[];
    emergencyContacts: string[];
    transportationArrangements: string;
  };
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [uploadRows, setUploadRows] = useState<{file: File | null, type: string}[]>([{ file: null, type: "Insurance" }]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  const [synthesisResult, setSynthesisResult] = useState<SynthesisResult | null>(null);
  const [generatingOrder, setGeneratingOrder] = useState(false);
  const [dischargeOrder, setDischargeOrder] = useState<DischargeOrder | null>(null);
  
  const docTypes = ["Insurance", "Progress Notes", "DME orders", "Nurse rounding", "Medical reports"];

  // Trigger synthesis when all results are complete
  useEffect(() => {
    const allComplete = results.length > 0 && results.every(r => !r.loading && !r.error && r.analysis);
    if (allComplete && !synthesizing && !synthesisResult) {
      handleSynthesis();
    }
  }, [results, synthesizing, synthesisResult]);

  const handleSynthesis = async () => {
    setSynthesizing(true);
    
    try {
      const response = await fetch('/api/synthesizer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analyses: results }),
      });
      
      const result = await response.json();
      setSynthesisResult(result);
    } catch (error) {
      console.error('Synthesis failed:', error);
    } finally {
      setSynthesizing(false);
    }
  };

  const handleCreateDischargeOrder = async () => {
    setGeneratingOrder(true);
    
    try {
      const response = await fetch('/api/discharge-order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analyses: results }),
      });
      
      const result = await response.json();
      setDischargeOrder(result.dischargeOrder);
    } catch (error) {
      console.error('Discharge order generation failed:', error);
    } finally {
      setGeneratingOrder(false);
    }
  };

  const handleDownloadPDF = async () => {
    const html2pdf = (await import('html2pdf.js')).default;
    const element = document.getElementById('discharge-order');
    
    const opt = {
      margin: 0.5,
      filename: `discharge-order-${new Date().toISOString().split('T')[0]}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { 
        scale: 2,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff'
      },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    
    // @ts-ignore
    html2pdf().set(opt).from(element).save();
  };

  const handleUpload = async () => {
    const validRows = uploadRows.filter(row => row.file);
    
    if (validRows.length === 0) return;
    
    const newFiles = validRows.map(row => row.file!);
    setFiles([...files, ...newFiles]);
    setShowModal(false);
    setUploadRows([{ file: null, type: "Insurance" }]);
    
    // Initialize results
    const initialResults = validRows.map(row => ({
      type: row.type,
      analysis: "",
      loading: true
    }));
    setResults(initialResults);
    setShowResults(true);
    
    // Process uploads
    for (const row of validRows) {
      const formData = new FormData();
      formData.append('file', row.file!);
      
      const endpoint = row.type.toLowerCase().replace(' ', '-');
      
      try {
        const response = await fetch(`/api/${endpoint}`, {
          method: 'POST',
          body: formData,
        });
        
        const result = await response.json();
        
        setResults(prev => prev.map(r => 
          r.type === row.type 
            ? { ...r, analysis: result.analysis, loading: false, error: response.ok ? undefined : result.error }
            : r
        ));
      } catch (error) {
        setResults(prev => prev.map(r => 
          r.type === row.type 
            ? { ...r, loading: false, error: 'Network error occurred' }
            : r
        ));
      }
    }
  };

  return (
    <div className={`min-h-screen flex flex-col items-center p-4 sm:p-8 bg-gradient-to-br from-gray-50 to-white ${showResults ? 'justify-start pt-12' : 'justify-center'}`}>
      {!showResults && (
        <>
          <h1 className="text-4xl sm:text-6xl font-light tracking-tight text-center mb-8 sm:mb-12 text-gray-800">
            Redefining Care with Intelligence
          </h1>
          
          <div className="flex flex-wrap justify-center gap-3 mb-8 max-w-4xl">
            {files.map((file, i) => (
              <div key={i} className="bg-white/60 backdrop-blur-xl border border-white/40 px-4 py-2 rounded-full text-gray-700 font-medium transition-all duration-200 hover:bg-white/70">
                {file.name}
              </div>
            ))}
          </div>

          <button 
            onClick={() => setShowModal(true)}
            className="bg-black/80 text-white px-8 py-3 rounded-2xl hover:bg-black/90 transition-all duration-200 backdrop-blur-xl hover:scale-105 font-medium"
          >
            Patient Readiness
          </button>
        </>
      )}

        {showModal && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <div className="bg-white/95 backdrop-blur-xl rounded-3xl p-4 w-full max-w-2xl border border-white/40">
              <h2 className="text-xl font-medium mb-6 text-gray-800">Upload Documents</h2>
              
              <div className="space-y-4 mb-6">
                {uploadRows.map((row, i) => (
                  <div key={i} className="flex gap-3">
                    <div className="flex-1 relative">
                      <input 
                        id={`file-${i}`}
                        type="file" 
                        accept=".pdf,.txt,.doc,.docx"
                        onChange={(e) => {
                          const newRows = [...uploadRows];
                          newRows[i].file = e.target.files?.[0] || null;
                          setUploadRows(newRows);
                        }}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                      <label 
                        htmlFor={`file-${i}`}
                        className="block bg-white/50 border border-white/40 rounded-xl px-3 py-2 text-sm backdrop-blur-sm cursor-pointer hover:bg-white/60 transition-colors"
                      >
                        {row.file ? row.file.name : "Choose file"}
                      </label>
                    </div>
                    <select 
                      value={row.type}
                      onChange={(e) => {
                        const newRows = [...uploadRows];
                        newRows[i].type = e.target.value;
                        setUploadRows(newRows);
                      }}
                      className="bg-white/50 border border-white/40 rounded-xl px-3 py-2 text-sm backdrop-blur-sm min-w-[140px]"
                    >
                      {docTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              <div className="flex justify-end mb-6">
                <button 
                  onClick={() => setUploadRows([...uploadRows, { file: null, type: "Insurance" }])}
                  className="w-10 h-10 bg-gray-500/80 text-white rounded-full flex items-center justify-center hover:bg-gray-600/80 transition-all duration-200 backdrop-blur-xl hover:scale-110 font-semibold text-lg"
                >
                  +
                </button>
              </div>

            <div className="flex gap-3">
              <button 
                onClick={() => setShowModal(false)}
                className="flex-1 bg-gray-200/80 backdrop-blur-xl px-4 py-3 rounded-xl hover:bg-gray-300/80 transition-all duration-200 font-medium"
              >
                Cancel
              </button>
              <button 
                onClick={handleUpload}
                className="flex-1 bg-black/80 text-white px-4 py-3 rounded-xl hover:bg-black/90 transition-all duration-200 backdrop-blur-xl font-medium"
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {showResults && (
        <div className="w-full max-w-6xl space-y-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-light text-gray-800">
              {synthesizing ? "Synthesizing Case..." : synthesisResult ? "Discharge Assessment" : "Analysis Pending..."}
            </h2>
            <button
              onClick={() => {
                setShowResults(false);
                setResults([]);
                setSynthesisResult(null);
                setSynthesizing(false);
                setDischargeOrder(null);
                setGeneratingOrder(false);
              }}
              className="bg-gray-500/80 text-white px-4 py-2 rounded-xl hover:bg-gray-600/80 transition-all duration-200 font-medium"
            >
              Back
            </button>
          </div>
          {results.map((result, i) => (
            <div key={i} className={`bg-white/60 backdrop-blur-xl border border-gray-200 rounded-2xl p-6 h-64 transition-all duration-1000 ${synthesizing ? 'animate-pulse' : ''}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-medium text-gray-800">{result.type}</h3>
                {result.loading && (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-600"></div>
                )}
              </div>
              
              <div className="h-44 overflow-y-auto">
                {result.loading ? (
                  <div className="space-y-3">
                    <div className="h-4 bg-gray-300/50 rounded animate-pulse"></div>
                    <div className="h-4 bg-gray-300/50 rounded w-3/4 animate-pulse"></div>
                    <div className="h-4 bg-gray-300/50 rounded w-1/2 animate-pulse"></div>
                  </div>
                ) : result.error ? (
                  <p className="text-red-600 font-medium">Error: {result.error}</p>
                ) : (
                  <p className="text-gray-700 leading-relaxed">{result.analysis}</p>
                )}
              </div>
            </div>
          ))}

          {synthesisResult && (
            <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6 mt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-blue-800">Discharge Recommendation</h3>
                <button
                  onClick={handleCreateDischargeOrder}
                  disabled={generatingOrder}
                  className="bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 transition-all duration-200 font-medium disabled:opacity-50"
                >
                  {generatingOrder ? "Generating..." : "Create Discharge Order"}
                </button>
              </div>
              <p className="text-blue-700 leading-relaxed">{synthesisResult.synthesis}</p>
            </div>
          )}

          {dischargeOrder && (
            <>
              <div className="flex justify-end mb-4">
                <button
                  onClick={handleDownloadPDF}
                  className="bg-green-600 text-white px-4 py-2 rounded-xl hover:bg-green-700 transition-all duration-200 font-medium"
                >
                  Download PDF
                </button>
              </div>
              <div id="discharge-order" className="bg-white border border-black rounded-lg p-8 font-mono text-sm shadow-lg">
                <div className="text-center mb-8">
                  <h2 className="text-3xl font-bold text-black mb-2">HOSPITAL DISCHARGE ORDER</h2>
                  <div className="w-full h-1 bg-black"></div>
                </div>
              
              {/* Patient Information */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 bg-gray-100 p-4 rounded border border-black">
                <div>
                  <p><strong>Patient Name:</strong> {dischargeOrder.patientName}</p>
                  <p><strong>MRN:</strong> {dischargeOrder.medicalRecordNumber}</p>
                </div>
                <div>
                  <p><strong>DOB:</strong> {dischargeOrder.dateOfBirth}</p>
                  <p><strong>Admission Date:</strong> {dischargeOrder.admissionDate}</p>
                </div>
                <div>
                  <p><strong>Discharge Date:</strong> {dischargeOrder.dischargeDate}</p>
                  <p><strong>Disposition:</strong> {dischargeOrder.dischargeDisposition}</p>
                </div>
              </div>

              {/* Diagnoses */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">PRIMARY DIAGNOSIS:</h4>
                <p className="bg-gray-100 p-2 rounded border-l-4 border-black">{dischargeOrder.primaryDiagnosis}</p>
              </div>

              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">SECONDARY DIAGNOSES:</h4>
                <ul className="list-decimal list-inside space-y-1 bg-gray-100 p-3 rounded border border-black">
                  {dischargeOrder.secondaryDiagnoses.map((diag, i) => <li key={i}>{diag}</li>)}
                </ul>
              </div>

              {/* Vital Signs */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">VITAL SIGNS AT DISCHARGE:</h4>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 bg-gray-100 p-3 rounded border border-black">
                  <p><strong>BP:</strong> {dischargeOrder.vitalSignsAtDischarge.bloodPressure}</p>
                  <p><strong>HR:</strong> {dischargeOrder.vitalSignsAtDischarge.heartRate}</p>
                  <p><strong>Temp:</strong> {dischargeOrder.vitalSignsAtDischarge.temperature}</p>
                  <p><strong>O2 Sat:</strong> {dischargeOrder.vitalSignsAtDischarge.oxygenSaturation}</p>
                  <p><strong>RR:</strong> {dischargeOrder.vitalSignsAtDischarge.respiratoryRate}</p>
                </div>
              </div>

              {/* Medications */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">DISCHARGE MEDICATIONS:</h4>
                <div className="space-y-3 bg-gray-100 p-3 rounded border border-black">
                  {dischargeOrder.medications.map((med, i) => (
                    <div key={i} className="border-b border-gray-400 pb-2 last:border-b-0">
                      <p><strong>{med.name}</strong> - {med.dosage}</p>
                      <p className="text-xs text-gray-700">Frequency: {med.frequency} | Duration: {med.duration}</p>
                      <p className="text-xs text-gray-700">Instructions: {med.instructions}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Activity & Diet */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <h4 className="font-bold mb-2 text-black">ACTIVITY RESTRICTIONS:</h4>
                  <p className="bg-gray-100 p-3 rounded border-l-4 border-black">{dischargeOrder.activityRestrictions}</p>
                </div>
                <div>
                  <h4 className="font-bold mb-2 text-black">DIET INSTRUCTIONS:</h4>
                  <p className="bg-gray-100 p-3 rounded border-l-4 border-black">{dischargeOrder.dietInstructions}</p>
                </div>
              </div>

              {/* Functional Status */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">FUNCTIONAL STATUS:</h4>
                <p className="bg-gray-100 p-3 rounded border-l-4 border-black">{dischargeOrder.functionalStatus}</p>
              </div>

              {/* Follow-up */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">FOLLOW-UP INSTRUCTIONS:</h4>
                <ul className="list-disc list-inside space-y-1 bg-gray-100 p-3 rounded border border-black">
                  {dischargeOrder.followUpInstructions.map((instruction, i) => <li key={i}>{instruction}</li>)}
                </ul>
              </div>

              {/* Warning Signs */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">RETURN TO HOSPITAL IF:</h4>
                <ul className="list-disc list-inside space-y-1 bg-gray-100 p-3 rounded border-l-4 border-black">
                  {dischargeOrder.warningSignsToReturn.map((sign, i) => <li key={i} className="text-black font-semibold">{sign}</li>)}
                </ul>
              </div>

              {/* Discharge Instructions */}
              <div className="mb-6">
                <h4 className="font-bold mb-2 text-black">ADDITIONAL DISCHARGE INSTRUCTIONS:</h4>
                <div className="bg-gray-100 p-4 rounded border border-black space-y-3">
                  {dischargeOrder.dischargeInstructions.woundCare && (
                    <div>
                      <p><strong>Wound Care:</strong> {dischargeOrder.dischargeInstructions.woundCare}</p>
                    </div>
                  )}
                  <div>
                    <p><strong>Equipment Needed:</strong></p>
                    <ul className="list-disc list-inside ml-4">
                      {dischargeOrder.dischargeInstructions.equipmentNeeded.map((item, i) => <li key={i}>{item}</li>)}
                    </ul>
                  </div>
                  <div>
                    <p><strong>Emergency Contacts:</strong></p>
                    <ul className="list-disc list-inside ml-4">
                      {dischargeOrder.dischargeInstructions.emergencyContacts.map((contact, i) => <li key={i}>{contact}</li>)}
                    </ul>
                  </div>
                  <p><strong>Transportation:</strong> {dischargeOrder.dischargeInstructions.transportationArrangements}</p>
                </div>
              </div>

              {/* Physician Signature */}
              <div className="mt-8 pt-6 border-t-2 border-black">
                <div className="flex justify-between items-end">
                  <div>
                    <p><strong>Discharging Physician:</strong></p>
                    <div className="w-64 h-0.5 bg-black mt-8"></div>
                    <p className="text-xs text-gray-700 mt-1">Physician Signature</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-700">
                      Document generated: {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
