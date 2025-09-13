"use client";
import { useState } from "react";

interface AnalysisResult {
  type: string;
  analysis: string;
  loading: boolean;
  error?: string;
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [uploadRows, setUploadRows] = useState<{file: File | null, type: string}[]>([{ file: null, type: "Insurance" }]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  
  const docTypes = ["Insurance", "Progress Notes", "DME orders", "Nurse rounding", "Medical reports"];

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
            <h2 className="text-2xl font-light text-gray-800">Analysis Pending...</h2>
            <button
              onClick={() => {
                setShowResults(false);
                setResults([]);
              }}
              className="bg-gray-500/80 text-white px-4 py-2 rounded-xl hover:bg-gray-600/80 transition-all duration-200 font-medium"
            >
              Back
            </button>
          </div>
          {results.map((result, i) => (
            <div key={i} className="bg-white/60 backdrop-blur-xl border border-gray-200 rounded-2xl p-6 h-64">
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
        </div>
      )}
    </div>
  );
}
