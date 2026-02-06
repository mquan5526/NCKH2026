import React, { useState } from "react";
import axios from "axios";

// Header
const Header = ({ darkMode }) => (
  <header
    className={`py-10 shadow-lg transition-colors duration-300 ${
      darkMode
        ? "bg-gradient-to-r from-gray-900 to-gray-800"
        : "bg-gradient-to-r from-amber-900 to-yellow-900"
    }`}
  >
    <div className="container mx-auto px-4 text-center">
      <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight text-white drop-shadow-md">
        Hệ thống Phát hiện Deepfake
      </h1>
      <p
        className={`mt-4 text-lg max-w-2xl mx-auto ${
          darkMode ? "text-gray-300" : "text-amber-100"
        }`}
      >
        Sử dụng trí tuệ nhân tạo để phân tích video và phát hiện nội dung giả
        mạo.
      </p>
    </div>
  </header>
);

// Footer
const Footer = ({ darkMode }) => (
  <footer
    className={`py-6 mt-12 text-center transition-colors duration-300 ${
      darkMode ? "bg-gray-900 text-gray-300" : "bg-amber-900 text-amber-200"
    }`}
  >
    <div className="container mx-auto px-4">
      <p className="text-sm opacity-80">
        Đồ án nghiên cứu phát hiện Deepfake Video
      </p>
      <p className="text-sm mt-1 opacity-80">
        ©{" "}
        <span
          className={`font-semibold ${
            darkMode ? "text-blue-400" : "text-amber-300"
          }`}
        >
          Nguyễn Dương Minh Quan - 2251012119
        </span>
      </p>
    </div>
  </footer>
);

// Upload Form
const UploadForm = ({
  file,
  loading,
  handleSubmit,
  handleFileChange,
  error,
  darkMode,
}) => (
  <div
    className={`p-8 rounded-2xl shadow-xl transition-all duration-300 ${
      darkMode
        ? "bg-gray-800 border border-gray-700 text-gray-200"
        : "bg-white border border-amber-200"
    }`}
  >
    <h2
      className={`text-2xl font-bold mb-6 ${
        darkMode ? "text-gray-100" : "text-amber-900"
      }`}
    >
      Tải Video lên
    </h2>
    <form onSubmit={handleSubmit} className="space-y-6">
      <label
        htmlFor="video-file"
        className={`flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 transform hover:scale-105 ${
          darkMode
            ? "border-gray-600 bg-gray-700 text-gray-300 hover:border-blue-400 hover:bg-gray-600"
            : "border-amber-300 bg-amber-50 text-amber-600 hover:border-amber-600 hover:bg-amber-100"
        }`}
      >
        <svg
          className={`w-16 h-16 mb-4 ${
            darkMode ? "text-gray-400" : "text-amber-400"
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
          ></path>
        </svg>
        <span
          className={`font-semibold ${
            darkMode ? "text-gray-200" : "text-amber-800"
          }`}
        >
          Kéo và thả video vào đây
        </span>
        <span className="text-sm">hoặc nhấp để chọn tệp</span>
        <input
          type="file"
          id="video-file"
          className="hidden"
          accept="video/*"
          onChange={handleFileChange}
        />
      </label>
      {file && (
        <div className="text-center text-sm">
          Đã chọn tệp:{" "}
          <strong>{file.name}</strong> ({(file.size / 1024 / 1024).toFixed(2)} MB)
        </div>
      )}
      <button
        type="submit"
        className={`w-full py-4 px-6 rounded-xl font-bold text-lg uppercase tracking-wide transition-all duration-300 transform ${
          loading || !file
            ? darkMode
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-amber-300 cursor-not-allowed"
            : darkMode
            ? "bg-gradient-to-r from-gray-700 to-gray-900 text-white hover:scale-105 hover:shadow-lg"
            : "bg-gradient-to-r from-amber-800 to-yellow-900 text-white hover:scale-105 hover:shadow-lg"
        }`}
        disabled={loading || !file}
      >
        {loading ? "⏳ Đang phân tích..." : "Phân tích"}
      </button>
    </form>
  </div>
);

// Loading Spinner
const LoadingSpinner = ({ darkMode }) => (
  <div className="text-center p-8 my-4">
    <div
      className={`w-16 h-16 border-4 rounded-full animate-spin mx-auto mb-4 ${
        darkMode
          ? "border-gray-600 border-t-blue-400"
          : "border-amber-200 border-t-amber-700"
      }`}
    ></div>
    <p
      className={`${
        darkMode ? "text-gray-300" : "text-amber-700"
      } font-medium`}
    >
      Đang xử lý video...
    </p>
  </div>
);

// Result
const ResultDisplay = ({ result, processingTime, darkMode }) => {
  if (!result) return null;
  const isFake = result.is_fake;
  const confidence = (result.confidence * 100).toFixed(2);
  return (
    <div
      className={`p-6 mt-8 rounded-xl border-l-4 font-semibold text-lg shadow-md ${
        isFake
          ? "bg-red-50 border-red-600 text-red-800"
          : darkMode
          ? "bg-gray-700 border-blue-400 text-gray-200"
          : "bg-amber-50 border-amber-700 text-amber-900"
      }`}
    >
      <h2 className="text-2xl font-bold mb-4">Kết quả phân tích</h2>
      <div className="text-center">
        <h3 className="text-3xl font-bold mb-2">
          {isFake ? "VIDEO GIẢ MẠO" : "✅ VIDEO THẬT"}
        </h3>
        {/* Comment phần Độ tin cậy */}
        {/* <p className="text-lg mt-2">Độ tin cậy: {confidence}%</p> */}
        {processingTime && (
          <p className="text-sm mt-1 opacity-80">
            ⏱️ Thời gian xử lý: {processingTime} giây
          </p>
        )}
      </div>
    </div>
  );
};

// App
const App = () => {
  // Mặc định chỉ dùng method-d
  const [method] = useState("method-d");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [processingTime, setProcessingTime] = useState(0);
  const [darkMode, setDarkMode] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
    setProcessingTime(0);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Vui lòng chọn video!");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);
    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append("video", file);
      // Flask nhận method là D (cố định)
      formData.append("method", "D");

      const response = await axios.post(
        "http://localhost:5000/api/detect",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      console.log("API Response:", response.data);

      setResult(response.data);
      const endTime = Date.now();
      setProcessingTime(((endTime - startTime) / 1000).toFixed(2));
    } catch (err) {
      console.error("API Error:", err);
      setError(
        err.response?.data?.error || "Đã xảy ra lỗi trong quá trình phân tích."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={`min-h-screen flex flex-col font-inter transition-colors duration-300 ${
        darkMode ? "bg-gray-900 text-gray-100" : "bg-amber-100 text-gray-900"
      }`}
    >
      <Header darkMode={darkMode} />

      {/* Toggle Dark Mode */}
      <div className="container mx-auto px-4 mt-4 text-right">
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="px-4 py-2 rounded-lg shadow bg-gray-700 text-white hover:bg-gray-600 transition"
        >
          {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}
        </button>
      </div>

      <main className="flex-1 py-12 px-4 flex justify-center items-start">
        <div className="container mx-auto max-w-3xl">
          {/* Không cần selector nữa, chỉ còn UploadForm */}
          <UploadForm
            file={file}
            loading={loading}
            handleFileChange={handleFileChange}
            handleSubmit={handleSubmit}
            error={error}
            darkMode={darkMode}
          />

          {loading && <LoadingSpinner darkMode={darkMode} />}

          {error && (
            <div className="mt-8 p-6 rounded-xl bg-red-50 border-l-4 border-red-600 text-red-800 font-semibold shadow-md">
              {error}
            </div>
          )}

          {result && (
            <ResultDisplay
              result={result}
              processingTime={processingTime}
              darkMode={darkMode}
            />
          )}
        </div>
      </main>

      <Footer darkMode={darkMode} />
    </div>
  );
};

export default App;
