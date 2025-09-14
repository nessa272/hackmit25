"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import ParticleBackground from "./components/ParticleBackground";

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
  const router = useRouter();
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Add glowing cursor effect only on client
  useEffect(() => {
    if (!isClient) return;

    // Add DM Serif Text font
    const fontLink = document.createElement('link');
    fontLink.href = 'https://fonts.googleapis.com/css2?family=DM+Serif+Text:ital@0;1&display=swap';
    fontLink.rel = 'stylesheet';
    document.head.appendChild(fontLink);

    const style = document.createElement('style');
    style.textContent = `
      body {
        cursor: none !important;
      }
      * {
        cursor: none !important;
      }
      .cursor-glow {
        position: fixed;
        width: 14px;
        height: 14px;
        background: rgba(188, 210, 238, 0.56);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        box-shadow: 0 0 10px rgba(96, 165, 250, 1), 0 0 30px rgba(96, 165, 250, 0.8), 0 0 45px rgba(96, 165, 250, 0.4);
        transition: transform 0.1s ease-out;
      }
      .dm-serif-text-regular {
        font-family: "DM Serif Text", serif;
        font-weight: 400;
        font-style: normal;
      }
      .dm-serif-text-regular-italic {
        font-family: "DM Serif Text", serif;
        font-weight: 400;
        font-style: italic;
      }
    `;
    document.head.appendChild(style);

    const cursor = document.createElement('div');
    cursor.className = 'cursor-glow';
    document.body.appendChild(cursor);

    const moveCursor = (e: MouseEvent) => {
      cursor.style.left = e.clientX - 7 + 'px';
      cursor.style.top = e.clientY - 7 + 'px';
    };

    document.addEventListener('mousemove', moveCursor);

    return () => {
      document.head.removeChild(style);
      document.head.removeChild(fontLink);
      document.removeEventListener('mousemove', moveCursor);
      if (cursor.parentNode) {
        cursor.parentNode.removeChild(cursor);
      }
    };
  }, [isClient]);

  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  const [synthesisResult, setSynthesisResult] = useState<SynthesisResult | null>(null);
  const [generatingOrder, setGeneratingOrder] = useState(false);
  const [dischargeOrder, setDischargeOrder] = useState<DischargeOrder | null>(null);
  

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


  return (
    <>
      {!showResults && <ParticleBackground />}
      <motion.div
        className={`min-h-screen flex flex-col items-center p-4 sm:p-8 ${showResults ? 'bg-gradient-to-br from-gray-50 to-white justify-start pt-12' : 'justify-center'} relative z-10`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        style={{ pointerEvents: "none" }}
      >
      <AnimatePresence>
        {!showResults && (
          <motion.div
            key="hero"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.6 }}
            className="flex flex-col items-center"
          >
             <motion.h1
              className="tracking-wide text-center mb-4 text-white drop-shadow-lg dm-serif-text-regular"
              style={{ fontSize: '40px' }}
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: -30 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <span className="text-white">InfiniCare </span>
            </motion.h1>
            <motion.h1
              className="text-4xl sm:text-6xl tracking-normal text-center mb-4 text-white drop-shadow-lg dm-serif-text-regular-italic"
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 2, delay: 2}}
            >
              <span className="text-white">Redefining Care with Intelligence</span>
            </motion.h1>

            <motion.button
              onClick={() => router.push('/transition')}
              className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-12 py-4 rounded-full hover:from-blue-600 hover:to-blue-700 transition-all duration-300 backdrop-blur-xl font-medium text-lg shadow-2xl border border-blue-400/30 dm-serif-text-regular"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 20 }}
              transition={{ duration: 2, delay: 0.6 }}
              whileHover={{ scale: 1.05, boxShadow: "0 25px 50px -12px rgba(59, 130, 246, 0.5)" }}
              whileTap={{ scale: 0.95 }}
              style={{ pointerEvents: "auto" }}
            >
              START
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>


      <AnimatePresence>
        {showResults && (
          <motion.div
            key="results"
            className="w-full max-w-6xl space-y-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.6 }}
          >
            <motion.div
              className="flex justify-between items-center mb-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <motion.h2
                className="text-2xl font-light text-gray-800"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                {synthesizing ? "Synthesizing Case..." : synthesisResult ? "Discharge Assessment" : "Analysis Pending..."}
              </motion.h2>
              <motion.button
                onClick={() => {
                  setShowResults(false);
                  setResults([]);
                  setSynthesisResult(null);
                  setSynthesizing(false);
                  setDischargeOrder(null);
                  setGeneratingOrder(false);
                }}
                className="bg-gray-500/80 text-white px-4 py-2 rounded-xl hover:bg-gray-600/80 transition-all duration-200 font-medium"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Back
              </motion.button>
            </motion.div>
          {results.map((result, i) => (
            <motion.div
              key={i}
              className={`bg-white/60 backdrop-blur-xl border border-gray-200 rounded-2xl p-6 h-64 transition-all duration-1000 ${synthesizing ? 'animate-pulse' : ''}`}
              initial={{ opacity: 0, y: 50, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.5, delay: i * 0.1 + 0.4 }}
              whileHover={{ scale: 1.02, y: -5 }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-medium text-gray-800">{result.type}</h3>
                {result.loading && (
                  <motion.div
                    className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-600"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </div>

              <div className="h-44 overflow-y-auto">
                {result.loading ? (
                  <div className="space-y-3">
                    <motion.div
                      className="h-4 bg-gray-300/50 rounded"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                    />
                    <motion.div
                      className="h-4 bg-gray-300/50 rounded w-3/4"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
                    />
                    <motion.div
                      className="h-4 bg-gray-300/50 rounded w-1/2"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut", delay: 0.4 }}
                    />
                  </div>
                ) : result.error ? (
                  <motion.p
                    className="text-red-600 font-medium"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    Error: {result.error}
                  </motion.p>
                ) : (
                  <motion.p
                    className="text-gray-700 leading-relaxed"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    {result.analysis}
                  </motion.p>
                )}
              </div>
            </motion.div>
          ))}

          <AnimatePresence>
            {synthesisResult && (
              <motion.div
                className="bg-blue-50 border border-blue-200 rounded-2xl p-6 mt-6"
                initial={{ opacity: 0, y: 30, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -30, scale: 0.95 }}
                transition={{ duration: 0.5, type: "spring", stiffness: 300, damping: 25 }}
              >
                <div className="flex items-center justify-between mb-4">
                  <motion.h3
                    className="text-xl font-semibold text-blue-800"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.1 }}
                  >
                    Discharge Recommendation
                  </motion.h3>
                  <motion.button
                    onClick={handleCreateDischargeOrder}
                    disabled={generatingOrder}
                    className="bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 transition-all duration-200 font-medium disabled:opacity-50"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.1 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {generatingOrder ? "Generating..." : "Create Discharge Order"}
                  </motion.button>
                </div>
                <motion.p
                  className="text-blue-700 leading-relaxed"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  {synthesisResult.synthesis}
                </motion.p>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {dischargeOrder && (
              <motion.div
                key="discharge-order"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -50 }}
                transition={{ duration: 0.6, type: "spring", stiffness: 300, damping: 25 }}
              >
                <motion.div
                  className="flex justify-end mb-4"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.2 }}
                >
                  <motion.button
                    onClick={handleDownloadPDF}
                    className="bg-green-600 text-white px-4 py-2 rounded-xl hover:bg-green-700 transition-all duration-200 font-medium"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Download PDF
                  </motion.button>
                </motion.div>
              <motion.div
                id="discharge-order"
                className="bg-white border border-black rounded-lg p-8 font-mono text-sm shadow-lg"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
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
              </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
    </>
  );
}
