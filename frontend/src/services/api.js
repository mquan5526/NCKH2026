import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const detectDeepfake = async (videoFile, method = 'A') => {
  const formData = new FormData();
  formData.append('video', videoFile);
  formData.append('method', method);

  const response = await axios.post(`${API_BASE_URL}/detect`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getAvailableMethods = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/methods`);
    return response.data;
  } catch (error) {
    console.error('API methods endpoint not available, using fallback');
    return [
      { code: 'A', name: 'Frame-wise Baseline', description: 'Phương pháp cơ bản' },
      { code: 'B', name: 'Frame CNN + Voting', description: 'Kết hợp voting' },
      { code: 'C', name: 'Feature Aggregation', description: 'Trích xuất đặc trưng' },
      { code: 'D', name: 'Patch-level + Temporal', description: 'Mô hình thời gian' }
    ];
  }
};