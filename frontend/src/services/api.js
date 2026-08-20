const API_BASE_URL = "http://localhost:8000";

/**
 * Generic API request helper.
 */
const apiRequest = async (endpoint, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let data = null;

  try {
    data = await response.json();
  } catch {
    // Response may not contain JSON.
  }

  if (!response.ok) {
    const errorMessage =
      data?.detail ||
      data?.message ||
      "API request failed.";

    throw new Error(errorMessage);
  }

  return data;
};

/**
 * Send a message to the chatbot.
 */
export const sendChatMessage = async (message) => {
  return apiRequest("/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
    }),
  });
};

/**
 * Predict customer churn.
 *
 * Backend endpoint:
 * POST /predict?model_name=random_forest
 *
 * The customer object is sent directly as the request body.
 */
export const predictCustomer = async (
  customer,
  modelName = "random_forest"
) => {
  const query = new URLSearchParams({
    model_name: modelName,
  });

  return apiRequest(`/predict?${query.toString()}`, {
    method: "POST",
    body: JSON.stringify(customer),
  });
};