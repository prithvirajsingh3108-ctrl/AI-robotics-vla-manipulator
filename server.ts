import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Initialize server-side Gemini client
let ai: GoogleGenAI | null = null;
let geminiCooldownUntil = 0;

if (process.env.GEMINI_API_KEY) {
  ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });
}

// REST endpoint to generate ROS-like planning model and simulated execution stream
app.post("/api/plan", async (req, res) => {
  const { command } = req.body;

  if (!command || typeof command !== "string") {
    return res.status(400).json({ error: "Invalid natural language command" });
  }

  // Pre-configured default database-based objects to align tables
  const defaultWorkspaceObjects = [
    { name: "red cube", color: "#ef4444", x: 80, y: 150, z: 10, shape: "cube" },
    { name: "blue box", color: "#3b82f6", x: -100, y: 180, z: 20, shape: "box" },
    { name: "green sphere", color: "#22c55e", x: 120, y: 120, z: 10, shape: "sphere" },
    { name: "yellow container", color: "#eab308", x: -60, y: 140, z: 20, shape: "box" },
    { name: "orange pyramid", color: "#f97316", x: 40, y: 190, z: 15, shape: "pyramid" }
  ];

  // Helper keyword parser in case there's no API key or a failure occurs
  const getSimulatedBrainPlan = (cmd: string) => {
    const cleaned = cmd.toLowerCase();
    
    let target = "red cube";
    let receptor = "blue box";
    
    if (cleaned.includes("green") && cleaned.includes("sphere")) {
      target = "green sphere";
    } else if (cleaned.includes("orange") || cleaned.includes("pyramid")) {
      target = "orange pyramid";
    } else if (cleaned.includes("red") || cleaned.includes("cube")) {
      target = "red cube";
    } else if (cleaned.includes("yellow") || cleaned.includes("container")) {
      target = "yellow container";
    }

    if (cleaned.includes("yellow") && (cleaned.includes("container") || cleaned.includes("box"))) {
      receptor = "yellow container";
    } else if (cleaned.includes("blue") || cleaned.includes("box")) {
      receptor = "blue box";
    } else if (cleaned.includes("red")) {
      receptor = "red cube"; // edge case
    }

    const targetObj = defaultWorkspaceObjects.find(o => o.name === target) || defaultWorkspaceObjects[0];
    const receptorObj = defaultWorkspaceObjects.find(o => o.name === receptor) || defaultWorkspaceObjects[1];

    return {
      understoodCommand: `Pick up the ${targetObj.name} and deposit it precisely in the ${receptorObj.name}.`,
      targetObject: targetObj.name,
      targetReceptor: receptorObj.name,
      detectedObjects: defaultWorkspaceObjects,
      planSteps: [
        {
          stepNum: 1,
          action: "locate",
          description: `Computer vision identifies coordinates of ${targetObj.name} and ${receptorObj.name} under camera frame.`,
          targetX: targetObj.x,
          targetY: targetObj.y,
          targetZ: targetObj.z
        },
        {
          stepNum: 2,
          action: "move_to",
          description: `Calculating joint angles via Inverse Kinematics. Moving arm gripper above the ${targetObj.name}.`,
          targetX: targetObj.x,
          targetY: targetObj.y,
          targetZ: targetObj.z + 50
        },
        {
          stepNum: 3,
          action: "grasp",
          description: `Gripper positioned. Actuating servo motors to secure grip with force feedback validation.`,
          targetX: targetObj.x,
          targetY: targetObj.y,
          targetZ: targetObj.z
        },
        {
          stepNum: 4,
          action: "lift",
          description: `Safely raising target object to clear obstacle height threshold (+80mm).`,
          targetX: targetObj.x,
          targetY: targetObj.y,
          targetZ: targetObj.z + 80
        },
        {
          stepNum: 5,
          action: "transport",
          description: `Path planning trajectory active. Moving arm smoothly toward the target ${receptorObj.name} receptor.`,
          targetX: receptorObj.x,
          targetY: receptorObj.y,
          targetZ: receptorObj.z + 60
        },
        {
          stepNum: 6,
          action: "release",
          description: `Receptor alignment successful. Releasing gripper payload and verifying object deposition.`,
          targetX: receptorObj.x,
          targetY: receptorObj.y,
          targetZ: receptorObj.z
        },
        {
          stepNum: 7,
          action: "home",
          description: `Returning robot manipulator to resting home configuration. Ready for next command loop.`,
          targetX: 0,
          targetY: 120,
          targetZ: 100
        }
      ]
    };
  };

  if (!ai || Date.now() < geminiCooldownUntil) {
    // Return high quality simulated plan aligned to SQLite defaults
    const simulated = getSimulatedBrainPlan(command);
    const msg = !ai 
      ? "Gemini API Key is missing. Running in local simulation mode." 
      : "Gemini API is currently experiencing high demand/spikes (503 Service Unavailable). Activated automatic 5-minute offline simulation safety cooldown.";
    return res.json({ ...simulated, mode: "fallback_simulation", error: msg });
  }

  try {
    const prompt = `You are the Vision-Language-Action (VLA) AI Brain of a 3-DOF simulated robotic arm manipulator.
The workstation table has a bounding box ranging from X [-150 to 150] and Y [100 to 220]. 
Here are the existing active objects and coordinates on the table workspace:
${JSON.stringify(defaultWorkspaceObjects, null, 2)}

Analyze this natural language command request from the operator:
"${command}"

Determine:
1. The physical 'targetObject' that needs to be picked up.
2. The 'targetReceptor' representing where it should be placed.
3. Formulate the path steps. The first step coordinates should match the targetObject's physical position.
4. Formulate the transportation step to match the targetReceptor's physical coordinates.
5. All detected objects must be preserved in the output list.

Output the full JSON plan strictly following the schema.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            understoodCommand: { 
              type: Type.STRING, 
              description: "Elegant rephrasing of operator's requested action." 
            },
            targetObject: { 
              type: Type.STRING, 
              description: "The name of the item to grasp." 
            },
            targetReceptor: { 
              type: Type.STRING, 
              description: "The name of the target place destination." 
            },
            detectedObjects: {
              type: Type.ARRAY,
              description: "Keep all default 5 objects but match the layout.",
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  color: { type: Type.STRING },
                  x: { type: Type.NUMBER },
                  y: { type: Type.NUMBER },
                  z: { type: Type.NUMBER },
                  shape: { type: Type.STRING }
                },
                required: ["name", "color", "x", "y", "z", "shape"]
              }
            },
            planSteps: {
              type: Type.ARRAY,
              description: "7 granular sequential planning steps to go from Home, pick-up, transport, deposit, to back Home.",
              items: {
                type: Type.OBJECT,
                properties: {
                  stepNum: { type: Type.INTEGER },
                  action: { type: Type.STRING, description: "Must be one of: locate, move_to, grasp, lift, transport, release, home" },
                  description: { type: Type.STRING, description: "Detailed visual and physical action happening." },
                  targetX: { type: Type.NUMBER },
                  targetY: { type: Type.NUMBER },
                  targetZ: { type: Type.NUMBER }
                },
                required: ["stepNum", "action", "description", "targetX", "targetY", "targetZ"]
              }
            }
          },
          required: ["understoodCommand", "targetObject", "targetReceptor", "detectedObjects", "planSteps"]
        }
      }
    });

    const parsedPlan = JSON.parse(response.text || "{}");
    return res.json({ ...parsedPlan, mode: "cognitive_ai" });

  } catch (error: any) {
    const errorStr = String(error?.message || JSON.stringify(error) || error || "");
    const isTransientOrQuota = errorStr.toLowerCase().includes("quota") ||
                               errorStr.toLowerCase().includes("exhausted") ||
                               errorStr.toLowerCase().includes("429") ||
                               errorStr.toLowerCase().includes("503") ||
                               errorStr.toLowerCase().includes("unavailable") ||
                               errorStr.toLowerCase().includes("high demand") ||
                               errorStr.toLowerCase().includes("spikes in demand") ||
                               errorStr.toLowerCase().includes("overloaded");
    if (isTransientOrQuota) {
      console.warn("Gemini API rate limited or experiencing transient overload. Running automatic local safety simulator fallback.");
      // Cooldown for 5 minutes to bypass further API calls
      geminiCooldownUntil = Date.now() + 5 * 60 * 1000;
    } else {
      console.error("Gemini Brain planning error:", errorStr);
    }
    // Graceful fallback to rich simulation engine on error
    const simulated = getSimulatedBrainPlan(command);
    return res.json({ ...simulated, mode: "fallback_simulation", error: errorStr });
  }
});

// Configure Vite middleware for development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`AI Robotics Brain Server running on http://localhost:${PORT}`);
  });
}

startServer();
