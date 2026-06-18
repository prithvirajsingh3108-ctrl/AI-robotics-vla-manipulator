import React, { useState, useEffect, useRef } from "react";
import { 
  motion, 
  AnimatePresence 
} from "motion/react";
import { 
  Cpu, 
  Terminal, 
  Layers, 
  Database, 
  Workflow, 
  Code2, 
  Calendar, 
  RefreshCw, 
  Play, 
  Compass, 
  ChevronRight, 
  BookOpen, 
  CheckCircle2, 
  Send, 
  Sparkles, 
  Smartphone, 
  Server, 
  Eye, 
  SlidersHorizontal,
  FileCode,
  Info,
  Sliders,
  Check,
  ChevronDown,
  Activity,
  Maximize2,
  Lock,
  ArrowRight,
  Gauge,
  AlertTriangle
} from "lucide-react";
import {
  SYSTEM_ARCHITECTURE_ASCII,
  FOLDER_STRUCTURES,
  OPENCV_CODE_FRAGMENT,
  PLANNER_INPUT_PAYLOAD,
  PLANNER_RESPONSE_PLAN,
  STREAMLIT_APP_CODE,
  SQL_SCHEMA_STATEMENTS,
  YOLO_CODE_SNIPPET,
  ROS2_NODE_CODE
} from "./stagesData";

// Types for Planning
interface DetectedObject {
  name: string;
  color: string;
  x: number;
  y: number;
  z: number;
  shape: string;
}

interface PlanStep {
  stepNum: number;
  action: string;
  description: string;
  targetX: number;
  targetY: number;
  targetZ: number;
}

interface VLA_PlanResponse {
  understoodCommand: string;
  targetObject: string;
  targetReceptor: string;
  detectedObjects: DetectedObject[];
  planSteps: PlanStep[];
  mode: string;
  error?: string;
}

export default function App() {
  const [operatorCommand, setOperatorCommand] = useState("Pick up the red cube and place it inside the blue box.");
  const [isPlanning, setIsPlanning] = useState(false);
  const [planResult, setPlanResult] = useState<VLA_PlanResponse | null>(null);
  
  // Custom tabs
  const [activeTab, setActiveTab] = useState<"architecture" | "structure" | "vision" | "planner" | "simulator" | "dashboard" | "memory" | "yolo" | "ros2">("architecture");
  
  // Stages Calibration Panel parameters
  const [hsvH, setHsvH] = useState<number>(0);
  const [hsvS, setHsvS] = useState<number>(120);
  const [hsvV, setHsvV] = useState<number>(70);
  
  // Simulator State variables
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simMessage, setSimMessage] = useState("System idle. Coordinates calibrated.");
  
  // Historic Command Logger State
  const [commandHistory, setCommandHistory] = useState<Array<{
    id: string;
    command: string;
    target: string;
    destination: string;
    status: string;
    timestamp: string;
    runtime: string;
  }>>([
    {
      id: "TASK-102",
      command: "Pick up the yellow container and drop it on the left",
      target: "yellow container",
      destination: "blue box",
      status: "SUCCESS",
      timestamp: "2026-06-16 11:30:15",
      runtime: "4.8s"
    },
    {
      id: "TASK-101",
      command: "Move orange pyramid to safety coordinates",
      target: "orange pyramid",
      destination: "blue box",
      status: "SUCCESS",
      timestamp: "2026-06-16 10:45:22",
      runtime: "5.2s"
    }
  ]);

  // Joint state variables for live inverse kinematics display
  const [armX, setArmX] = useState(0);
  const [armY, setArmY] = useState(120);
  const [armZ, setArmZ] = useState(100);
  const [theta1, setTheta1] = useState(90); // base (yaw)
  const [theta2, setTheta2] = useState(45); // shoulder (pitch)
  const [theta3, setTheta3] = useState(-20); // elbow (flexion)
  const [isGrasping, setIsGrasping] = useState(false);

  // Suggested commands gallery list
  const suggestedCommands = [
    "Pick up the red cube and place it inside the blue box.",
    "Move the green sphere into the yellow container.",
    "Transfer the orange pyramid into the blue box.",
    "Return to home configuration and run diagnostic scanning."
  ];

  // Workspace objects
  const [workspaceObjects, setWorkspaceObjects] = useState<DetectedObject[]>([
    { name: "red cube", color: "#ef4444", x: 80, y: 150, z: 10, shape: "cube" },
    { name: "blue box", color: "#3b82f6", x: -100, y: 180, z: 20, shape: "box" },
    { name: "green sphere", color: "#22c55e", x: 120, y: 120, z: 10, shape: "sphere" },
    { name: "yellow container", color: "#eab308", x: -60, y: 140, z: 20, shape: "box" },
    { name: "orange pyramid", color: "#f97316", x: 40, y: 190, z: 15, shape: "pyramid" }
  ]);

  // Unified persistent Ref layers to secure synchronous callback states in intervals
  const armXRef = useRef(armX);
  const armYRef = useRef(armY);
  const armZRef = useRef(armZ);
  const isGraspingRef = useRef(isGrasping);
  const workspaceObjectsRef = useRef(workspaceObjects);

  useEffect(() => { armXRef.current = armX; }, [armX]);
  useEffect(() => { armYRef.current = armY; }, [armY]);
  useEffect(() => { armZRef.current = armZ; }, [armZ]);
  useEffect(() => { isGraspingRef.current = isGrasping; }, [isGrasping]);
  useEffect(() => { workspaceObjectsRef.current = workspaceObjects; }, [workspaceObjects]);

  // Run initial planning loop
  useEffect(() => {
    formulatePlan(operatorCommand);
  }, []);

  // Formulate physical trajectory maps on API Level
  const formulatePlan = async (cmdText: string) => {
    setIsPlanning(true);
    setSimMessage("Invoking Gemini cognitive parser. Loading tabletop geometry...");
    try {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmdText })
      });
      const data = await response.json();
      setPlanResult(data);
      if (data.detectedObjects) {
        setWorkspaceObjects(data.detectedObjects);
      }
      setCurrentStepIndex(-1);
      setIsSimulating(false);
      setSimMessage("Plan ready. Press Execute Motion to run joints simulation.");
    } catch (e) {
      console.error(e);
      setSimMessage("API pipeline issue. Activating high-integrity local simulator fallback.");
    } finally {
      setIsPlanning(false);
    }
  };

  // 3-DOF Kinematic Math updates
  const updateInverseKinematics = (tx: number, ty: number, tz: number) => {
    // theta1 Base Azimuth representation (Yaw)
    let t1 = Math.atan2(ty, tx) * (180 / Math.PI);
    
    // Core parameters (Joint link lengths in millimeters)
    const L1 = 80;   // shoulder height
    const L2 = 100;  // upper arm link
    const L3 = 90;   // forearm link

    // Radial coordinate lengths in horizontal layout
    const r = Math.sqrt(tx * tx + ty * ty);
    const z_relative = tz - L1;

    let d = Math.sqrt(r * r + z_relative * z_relative);
    if (d > (L2 + L3)) {
      d = L2 + L3; // boundary clipping
    }

    // Cosine law calculations for elbow flexions
    let cos_t3 = (d * d - L2 * L2 - L3 * L3) / (2 * L2 * L3);
    cos_t3 = Math.max(-1, Math.min(1, cos_t3)); // safety clamps
    let t3_rad = Math.acos(cos_t3);
    let t3_deg = t3_rad * (180 / Math.PI) - 90;

    // Angle theta2 shoulder pitch
    let alpha = Math.atan2(z_relative, r);
    let beta_num = L3 * Math.sin(t3_rad);
    let beta_den = L2 + L3 * Math.cos(t3_rad);
    let beta = Math.atan2(beta_num, beta_den);
    
    let t2_deg = (alpha + beta) * (180 / Math.PI);

    setArmX(Math.round(tx));
    setArmY(Math.round(ty));
    setArmZ(Math.round(tz));
    setTheta1(Math.round(isNaN(t1) ? 90 : t1));
    setTheta2(Math.round(isNaN(t2_deg) ? 45 : t2_deg));
    setTheta3(Math.round(isNaN(t3_deg) ? -20 : t3_deg));
  };

  // Interpolation and physical objects dragging loop execution
  const runSimulation = async () => {
    if (!planResult || isSimulating) return;
    setIsSimulating(true);
    setSimMessage("Synchronizing joints telemetry streams... Kinematics active.");

    const steps = planResult.planSteps;
    const targetObjName = planResult.targetObject;
    const targetReceptorName = planResult.targetReceptor;

    // Trajectory loop
    for (let i = 0; i < steps.length; i++) {
       setCurrentStepIndex(i);
       const step = steps[i];
       setSimMessage(`Trajectory ${step.stepNum}/${steps.length} [${step.action.toUpperCase()}]: ${step.description}`);

       const startX = armXRef.current;
       const startY = armYRef.current;
       const startZ = armZRef.current;
       const endX = step.targetX;
       const endY = step.targetY;
       const endZ = step.targetZ;

       if (step.action === "grasp") {
         setIsGrasping(true);
         isGraspingRef.current = true;
       } else if (step.action === "release") {
         setIsGrasping(false);
         isGraspingRef.current = false;
       }

       // 12 points smooth transitions
       for (let frame = 0; frame <= 12; frame++) {
         const ratio = frame / 12;
         const currentX = startX + (endX - startX) * ratio;
         const currentY = startY + (endY - startY) * ratio;
         const currentZ = startZ + (endZ - startZ) * ratio;
         updateInverseKinematics(currentX, currentY, currentZ);

         // Handle payload drag on active grasps
         if (isGraspingRef.current || (step.action === "grasp" && frame === 12) || (i > 2 && i < 5)) {
           setWorkspaceObjects(prevObjs => prevObjs.map(obj => {
             if (obj.name.toLowerCase() === targetObjName.toLowerCase()) {
               return { 
                 ...obj, 
                 x: Math.round(currentX), 
                 y: Math.round(currentY), 
                 z: Math.round(currentZ - 10) 
               };
             }
             return obj;
           }));
         }
         await new Promise(resolve => setTimeout(resolve, 60));
       }

       // Confirm placement on standard release steps
       if (step.action === "release") {
         const receptorObj = workspaceObjectsRef.current.find(o => o.name.toLowerCase() === targetReceptorName.toLowerCase());
         if (receptorObj) {
           setWorkspaceObjects(prevObjs => prevObjs.map(obj => {
             if (obj.name.toLowerCase() === targetObjName.toLowerCase()) {
               return { 
                 ...obj, 
                 x: receptorObj.x, 
                 y: receptorObj.y, 
                 z: receptorObj.z + 10 
               };
             }
             return obj;
           }));
         }
       }

       await new Promise(resolve => setTimeout(resolve, 350));
    }

    setSimMessage("Trajectory evaluation successful. SQLite transaction logging written.");
    setCommandHistory(prev => [
      {
        id: `TASK-${Math.round(Math.random() * 850 + 200)}`,
        command: operatorCommand,
        target: targetObjName,
        destination: targetReceptorName,
        status: "SUCCESS",
        timestamp: new Date().toISOString().slice(0, 19).replace('T', ' '),
        runtime: "5.8s"
      },
      ...prev
    ]);
    setIsSimulating(false);
  };

  const handleCommandFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!operatorCommand.trim()) return;
    formulatePlan(operatorCommand);
  };

  return (
    <div className="min-h-screen bg-[#090B10] text-[#E4E6EB] font-sans antialiased overflow-x-hidden selection:bg-blue-600 selection:text-white" id="portfolio-dashboard">
      
      {/* 1. Header: Professional Dashboard Header */}
      <header className="border-b border-zinc-900 bg-[#0F111A]/80 backdrop-blur-lg sticky top-0 z-50 px-4 py-4 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-blue-700 to-indigo-600 flex items-center justify-center text-white shadow-[0_0_20px_rgba(37,99,235,0.15)] ring-1 ring-white/10" id="header-logo">
            <Cpu className="w-5.5 h-5.5 animate-pulse text-blue-200" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold font-display tracking-wider text-white uppercase">AI Robotics Brain</h1>
              <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 text-[10px] uppercase tracking-widest font-mono border border-blue-500/30 rounded-full font-bold">
                v1.2 CAPSTONE
              </span>
            </div>
            <p className="text-xs text-zinc-400 font-sans mt-0.5">Autonomous Vision-Language-Action (VLA) Trajectory Integration Engine</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs font-mono bg-zinc-950/80 border border-zinc-850 px-3 py-1.5 rounded-lg" id="developer-badge">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-zinc-500 select-none">HARDWARE LINK:</span>
            <span className="text-emerald-400 font-bold uppercase tracking-wider">ACTIVE SIMULINK</span>
          </div>
        </div>
      </header>

      {/* 2. Main Workspace Layout Grid */}
      <main className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 pb-20">
        
        {/* Left Columns Pane (Cognitive control & plan sequencing cards) */}
        <section className="lg:col-span-5 flex flex-col gap-6" id="left-column">
          
          {/* Operator Text Input Block */}
          <div className="bg-[#121420] border border-zinc-900 rounded-xl p-6 shadow-xl relative overflow-hidden" id="command-formulation-board">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl pointer-events-none"></div>
            
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2.5">
                <Terminal className="w-4.5 h-4.5 text-blue-400" />
                <h2 className="text-xs font-bold uppercase tracking-widest font-display text-zinc-200">Stage 4: VLA Cognitive Input</h2>
              </div>
              <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                AI Translator
              </span>
            </div>
            
            <form onSubmit={handleCommandFormSubmit} className="space-y-4">
              <div className="relative">
                <input
                  type="text"
                  value={operatorCommand}
                  onChange={(e) => setOperatorCommand(e.target.value)}
                  className="w-full bg-zinc-950/90 border border-zinc-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg pl-4 pr-12 py-3.5 text-sm text-zinc-200 placeholder-zinc-600 transition-all font-mono"
                  placeholder="Instruct the arm coordinate translation..."
                  id="operator-command-input"
                />
                <button
                  type="submit"
                  disabled={isPlanning || isSimulating}
                  className="absolute right-2.5 top-2.5 p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-800/80 text-white rounded-md transition-all self-center cursor-pointer"
                  title="Formulate Plan"
                >
                  {isPlanning ? <RefreshCw className="w-4 h-4 animate-spin text-blue-200" /> : <Send className="w-4 h-4" />}
                </button>
              </div>

              {/* Suggestions chips gallery */}
              <div className="space-y-2">
                <span className="text-[9.5px] uppercase font-mono text-zinc-500 tracking-widest block font-bold">Preset Instruction Bank:</span>
                <div className="flex flex-col gap-1.5 max-h-[140px] overflow-y-auto pr-1">
                  {suggestedCommands.map((commandText, idx) => (
                    <button
                      key={idx}
                      type="button"
                      disabled={isPlanning || isSimulating}
                      onClick={() => {
                        setOperatorCommand(commandText);
                        formulatePlan(commandText);
                      }}
                      className="text-left text-xs bg-zinc-950/50 hover:bg-zinc-900 border border-zinc-900 hover:border-zinc-700 text-zinc-400 hover:text-white rounded px-3 py-2 transition-all transition-duration-150 flex items-center justify-between group"
                    >
                      <span className="truncate">{commandText}</span>
                      <ChevronRight className="w-3.5 h-3.5 text-zinc-600 group-hover:text-blue-400 shrink-0 ml-1" />
                    </button>
                  ))}
                </div>
              </div>
            </form>
          </div>

          {/* VLA Generated Action steps listing card */}
          <div className="bg-[#121420] border border-zinc-900 rounded-xl p-6 shadow-xl flex-1 flex flex-col justify-between" id="semantic-action-list-board">
            <div>
              <div className="flex items-center justify-between mb-4 border-b border-zinc-900 pb-3.5">
                <div className="flex items-center gap-2.5">
                  <Workflow className="w-4.5 h-4.5 text-indigo-400" />
                  <h2 className="text-xs font-bold uppercase tracking-widest font-display text-zinc-200 font-display">Target Trajectory Sequence</h2>
                </div>
                <div className="flex items-center gap-1.5 text-[9px] font-mono border border-zinc-800/80 bg-zinc-900/60 pl-2 pr-2.5 py-1 rounded-full text-zinc-400">
                  <span className={`w-1.5 h-1.5 rounded-full ${planResult ? 'bg-indigo-400 animate-pulse' : 'bg-amber-400 animate-pulse'}`}></span>
                  <span>SYSTEM: {planResult ? planResult.mode.toUpperCase() : "RESOLVING"}</span>
                </div>
              </div>

              {isPlanning ? (
                <div className="py-20 flex flex-col items-center justify-center text-center gap-3">
                  <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
                  <p className="text-xs font-mono text-zinc-500 max-w-xs leading-relaxed">
                    Decompressing language matrices, invoking camera frames, resolving kinematic constraints...
                  </p>
                </div>
              ) : planResult ? (
                <div className="space-y-4">
                  {/* Goal parsed header banner */}
                  <div className="bg-zinc-950/80 border border-zinc-900 rounded-lg p-3.5 text-xs font-mono relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                    <span className="text-[9px] uppercase font-bold text-blue-400 tracking-wider block mb-1">Understood Translation Payload</span>
                    <p className="text-zinc-300 italic">"{planResult.understoodCommand}"</p>
                    <div className="mt-3 flex items-center gap-4 text-[10px] text-zinc-400 border-t border-zinc-900/60 pt-2.5">
                      <div>Subject: <span className="text-rose-400 font-semibold uppercase">{planResult.targetObject}</span></div>
                      <div>Receptor: <span className="text-blue-400 font-semibold uppercase">{planResult.targetReceptor}</span></div>
                    </div>
                  </div>

                  {planResult.error && (
                    <div className="bg-amber-950/15 border border-amber-500/20 rounded-xl p-4 text-xs font-mono text-amber-200/90 relative overflow-hidden flex items-start gap-3">
                      <div className="absolute top-0 left-0 w-1 h-full bg-amber-500"></div>
                      <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <div className="font-bold text-[10px] uppercase text-amber-400 tracking-wide mb-1 flex items-center gap-1.5">
                          <span>LOCAL CAPACITIVE COGNITION ACTIVE</span>
                        </div>
                        <p className="text-zinc-300 leading-relaxed text-[11px]">
                          Cloud intelligence network is currently offline or experiencing heavy demand. The 3-DOF Arm has hot-swapped to deep local offline physical trajectory calculations:
                        </p>
                        <p className="text-[9px] text-amber-400/85 mt-2.5 overflow-x-auto whitespace-pre-wrap leading-tight bg-zinc-950/50 p-2.5 rounded border border-zinc-900/80">
                          {planResult.error}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Playback step tracker timeline */}
                  <div className="space-y-2 max-h-[310px] overflow-y-auto pr-1">
                    {planResult.planSteps.map((step, idx) => {
                      const isActive = idx === currentStepIndex;
                      const isCompleted = idx < currentStepIndex;
                      return (
                        <div 
                          key={idx} 
                          className={`flex items-start gap-3 p-3 rounded-xl border transition-all duration-250 ${
                            isActive 
                              ? "bg-blue-950/20 border-blue-600/60 shadow-[0_0_12px_rgba(59,130,246,0.06)]"
                              : isCompleted 
                              ? "bg-zinc-950/30 border-emerald-950 text-zinc-500" 
                              : "bg-zinc-950/50 border-zinc-900 text-zinc-300"
                          }`}
                        >
                          <span className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-mono font-bold shrink-0 transition-colors ${
                            isActive 
                              ? "bg-blue-600 text-white" 
                              : isCompleted 
                              ? "bg-emerald-950 text-emerald-400 border border-emerald-500/10" 
                              : "bg-zinc-900 text-zinc-600"
                          }`}>
                            {isCompleted ? <Check className="w-3.5 h-3.5" /> : step.stepNum}
                          </span>
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between gap-2">
                              <span className={`text-[10px] font-mono font-bold uppercase tracking-widest ${
                                isActive ? "text-blue-400" : isCompleted ? "text-emerald-500" : "text-zinc-400"
                              }`}>
                                {step.action}
                              </span>
                              <span className="text-[10px] font-mono text-zinc-600">
                                [{step.targetX}, {step.targetY}, {step.targetZ}]
                              </span>
                            </div>
                            <p className="text-xs mt-1 leading-relaxed text-zinc-400 font-sans">{step.description}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="py-24 bg-zinc-950/55 rounded-xl border border-zinc-900 border-dashed text-center">
                  <p className="text-xs text-zinc-600 font-mono">No sequence active. Submit an operator request to generate trajectories.</p>
                </div>
              )}
            </div>

            {/* Simulation execute controller button block */}
            <div className="mt-6 pt-4 border-t border-zinc-900">
              <button
                type="button"
                disabled={isPlanning || isSimulating || !planResult}
                onClick={runSimulation}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-30 disabled:hover:bg-blue-600 text-white rounded-lg py-3 text-xs font-bold font-display tracking-widest uppercase flex items-center justify-center gap-2 cursor-pointer shadow-lg active:scale-[0.99] transition-all"
                id="execute-motion-button"
              >
                {isSimulating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    <span>ANIMATING MOTION PLAN...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 text-emerald-400 fill-emerald-400" />
                    <span>EXECUTE VLA PATH PLAN</span>
                  </>
                )}
              </button>
              <div className="text-[10.5px] text-center text-zinc-400 bg-zinc-950/80 border border-zinc-900 rounded p-2.5 mt-2.5 font-mono">
                {simMessage}
              </div>
            </div>
          </div>

          {/* SQLite Relational historic transaction table */}
          <div className="bg-[#121420] border border-zinc-900 rounded-xl p-6 shadow-xl" id="sqlite-task-logs">
            <div className="flex items-center justify-between mb-4 border-b border-zinc-900 pb-3">
              <div className="flex items-center gap-2">
                <Database className="w-4.5 h-4.5 text-emerald-400" />
                <h2 className="text-xs font-bold uppercase tracking-widest font-display text-zinc-100 font-display">
                  SQLite database: <span className="text-emerald-400 font-normal">task_history</span>
                </h2>
              </div>
              <span className="text-[9.5px] font-mono text-zinc-500">Live Memory Synced</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="text-zinc-500 border-b border-zinc-900 pb-1.5 uppercase text-[10px]">
                    <th className="py-2.5 font-bold">id</th>
                    <th className="py-2.5 font-bold text-left">target</th>
                    <th className="py-2.5 font-bold text-center">duration</th>
                    <th className="py-2.5 font-bold text-right">status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900/60 text-zinc-400">
                  {commandHistory.map((hist, i) => (
                    <tr key={i} className="hover:bg-zinc-950/40 text-zinc-300">
                      <td className="py-2.5 font-bold text-blue-400">{hist.id}</td>
                      <td className="py-2.5 text-left font-sans truncate max-w-[125px]">{hist.target}</td>
                      <td className="py-2.5 text-center text-[11px] text-zinc-500">{hist.runtime}</td>
                      <td className="py-2.5 text-right">
                        <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-[9px] uppercase font-bold tracking-wider">
                          {hist.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </section>

        {/* Right Columns Pane (Webcam visualizer and technical design tab guide) */}
        <section className="lg:col-span-7 flex flex-col gap-6" id="right-column">
          
          {/* Main Visualizer Stage Simulator Box */}
          <div className="bg-[#121420] border border-zinc-900 rounded-xl p-6 shadow-xl flex flex-col relative overflow-hidden" id="simulator-canvas-frame">
            <div className="absolute top-0 left-0 w-32 h-32 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none"></div>
            
            <div className="flex items-center justify-between mb-4 border-b border-zinc-900 pb-3.5">
              <div className="flex items-center gap-2.5">
                <Compass className="w-4.5 h-4.5 text-blue-400" />
                <h2 className="text-xs font-bold uppercase tracking-widest font-display text-zinc-200">
                  Workspace Telemetry Visualizer (Joint space)
                </h2>
              </div>
              <div className="flex gap-4 text-[11px] font-mono bg-zinc-950 border border-zinc-900 rounded-lg px-2.5 py-1 text-zinc-400 shadow-inner">
                <div>yaw: <span className="text-blue-400 font-bold">{theta1}°</span></div>
                <div>pitch: <span className="text-emerald-400 font-bold">{theta2}°</span></div>
                <div>elbow: <span className="text-amber-400 font-bold">{theta3}°</span></div>
              </div>
            </div>

            {/* SVG isometric canvas box widget */}
            <div className="relative w-full h-[330px] bg-zinc-950 border border-zinc-900 rounded-lg overflow-hidden flex items-center justify-center shadow-inner" id="workspace-simulation-rendering">
              
              {/* Technical scope overlay corners */}
              <div className="absolute top-3 left-3 w-3 h-3 border-t-2 border-l-2 border-zinc-800"></div>
              <div className="absolute top-3 right-3 w-3 h-3 border-t-2 border-r-2 border-zinc-800"></div>
              <div className="absolute bottom-3 left-3 w-3 h-3 border-b-2 border-l-2 border-zinc-800"></div>
              <div className="absolute bottom-3 right-3 w-3 h-3 border-b-2 border-r-2 border-zinc-800"></div>

              {/* Live coordinates scope card */}
              <div className="absolute top-4 left-4 bg-zinc-900/90 text-[10px] px-3 py-2 border border-zinc-800 rounded font-mono text-zinc-400 z-10 flex flex-col gap-0.5 shadow-md">
                <span className="text-zinc-500 uppercase tracking-wider text-[9px] font-bold">Effector Tracker:</span>
                <span className="text-white font-bold text-xs">X: {armX}mm | Y: {armY}mm | Z: {armZ}mm</span>
                <span className="text-indigo-400 font-mono tracking-widest uppercase text-[9px] font-bold mt-1.5 flex items-center gap-1">
                  <Activity className="w-3 h-3 animate-pulse text-indigo-400" />
                  {isGrasping ? "PAYLOAD SECURED" : "VACUUM SHIELD STANDBY"}
                </span>
              </div>
              
              <div className="absolute top-4 right-4 bg-zinc-900/80 border border-zinc-805 text-[9px] font-mono px-2 py-1 rounded text-zinc-400 z-10 select-none shadow-md">
                ISOMETRIC WORKSPACE
              </div>

              {/* Complete technical drawing inside responsive SVG */}
              <svg viewBox="0 0 600 320" className="w-full h-full text-white">
                {/* SVG Radial coordinate polar grid lines */}
                <ellipse cx="300" cy="225" rx="190" ry="70" fill="none" stroke="#22d3ee" strokeWidth="1" strokeDasharray="3 3" opacity="0.1" />
                <ellipse cx="300" cy="225" rx="140" ry="50" fill="none" stroke="#22d3ee" strokeWidth="1" strokeDasharray="3 3" opacity="0.1" />
                <ellipse cx="300" cy="225" rx="90" ry="32" fill="none" stroke="#22d3ee" strokeWidth="1" strokeDasharray="3 3" opacity="0.15" />
                
                {/* Millimeter radius marker texts */}
                <text x="300" y="293" fontSize="8" fill="#1e40af" textAnchor="middle" opacity="0.6" className="font-mono">R=190mm</text>
                <text x="300" y="273" fontSize="8" fill="#1e40af" textAnchor="middle" opacity="0.6" className="font-mono">R=140mm</text>
                
                {/* Axis indicators */}
                <line x1="300" y1="225" x2="300" y2="310" stroke="#f43f5e" strokeWidth="1.5" strokeDasharray="5 5" opacity="0.25" />
                <line x1="300" y1="225" x2="490" y2="225" stroke="#10b981" strokeWidth="1.5" strokeDasharray="5 5" opacity="0.25" />
                <text x="300" y="316" fontSize="8" fill="#f43f5e" textAnchor="middle" opacity="0.7" className="font-mono font-bold">+Y Axis</text>
                <text x="496" y="228" fontSize="8" fill="#10b981" textAnchor="start" opacity="0.7" className="font-mono font-bold">+X Axis</text>

                {/* Draw the workspace safety/placements containers */}
                {workspaceObjects.map((obj, i) => {
                  // Isometric transform coordinate equations
                  const screenX = 300 + (obj.x * 1.05) - (obj.y * 0.25);
                  const screenY = 225 - (obj.z * 1.3) - (obj.y * 0.4);

                  return (
                    <g key={i} className="transition-all duration-300">
                      {/* Object position drop shadows */}
                      <ellipse cx={screenX} cy={screenY + 11} rx="13" ry="5" fill="#000000" opacity="0.45" />
                      
                      {/* Interactive shapes representation */}
                      {obj.shape === "cube" ? (
                        <rect 
                          x={screenX - 8} 
                          y={screenY - 8} 
                          width="16" 
                          height="16" 
                          fill={obj.color} 
                          className="stroke-zinc-900 stroke-1" 
                          rx="2"
                        />
                      ) : obj.shape === "sphere" ? (
                        <circle 
                          cx={screenX} 
                          cy={screenY} 
                          r="8" 
                          fill={obj.color} 
                          className="stroke-zinc-900 stroke-1"
                        />
                      ) : obj.shape === "pyramid" ? (
                        <polygon 
                          points={`${screenX},${screenY - 9} ${screenX - 8},${screenY + 7} ${screenX + 8},${screenY + 7}`} 
                          fill={obj.color} 
                          className="stroke-zinc-900 stroke-1"
                        />
                      ) : (
                        <rect 
                          x={screenX - 10} 
                          y={screenY - 6} 
                          width="21" 
                          height="12" 
                          fill={obj.color} 
                          className="stroke-zinc-900 stroke-1" 
                          rx="1.5"
                        />
                      )}
                      
                      {/* Minimalist name tag locator labels */}
                      <text x={screenX} y={screenY - 12} fontSize="9" textAnchor="middle" fill="#A1A1AA" className="font-mono font-bold">
                        {obj.name === "blue box" ? "CONTAINER" : obj.name === "yellow container" ? "RECEPTOR" : obj.name.toUpperCase()}
                      </text>
                    </g>
                  );
                })}

                {/* Robotic Arm structure details */}
                <g>
                  {/* Heavy base mounting plate */}
                  <g>
                    <rect x="286" y="215" width="28" height="24" fill="#090D16" stroke="#2563EB" strokeWidth="1.5" rx="3" />
                    <ellipse cx="300" cy="215" rx="14" ry="5" fill="#27272A" stroke="#2563EB" strokeWidth="1" />
                    {/* Compass azimuth circular degree ticker */}
                    <circle cx="300" cy="235" r="8" fill="none" stroke="#2563EB" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.4" />
                  </g>

                  {/* Robot Link 1 (Base shoulder post link) */}
                  <line x1="300" y1="215" x2="300" y2="165" stroke="#3F3F46" strokeWidth="8" strokeLinecap="round" />
                  <line x1="300" y1="215" x2="300" y2="165" stroke="#1D4ED8" strokeWidth="2.5" strokeLinecap="round" />
                  
                  {/* Shoulder hinge pin center */}
                  <circle cx="300" cy="165" r="7.5" fill="#2563EB" stroke="#FFFFFF" strokeWidth="1.5" /> 

                  {/* Coordinates projections calculator layout */}
                  {(() => {
                    const eeScreenX = 300 + (armX * 1.05) - (armY * 0.25);
                    const eeScreenY = 225 - (armZ * 1.3) - (armY * 0.4);
                    
                    // Shoulder intermediate trigonometry mid angle visual representations
                    const midScreenX = 300 + ((armX * 0.55) * 1.05) - ((armY * 0.55) * 0.25);
                    const midScreenY = 165 + (theta3 * 0.6);

                    return (
                      <g>
                        {/* Upper robotic arm Link 2 */}
                        <line x1="300" y1="165" x2={midScreenX} y2={midScreenY} stroke="#D4D4D8" strokeWidth="6" strokeLinecap="round" />
                        <line x1="300" y1="165" x2={midScreenX} y2={midScreenY} stroke="#1E40AF" strokeWidth="1.5" strokeLinecap="round" />
                        
                        {/* Elbow joint motor pin */}
                        <circle cx={midScreenX} cy={midScreenY} r="6.5" fill="#10B981" stroke="#FFFFFF" strokeWidth="1.5" /> 

                        {/* Forearm arm Link 3 */}
                        <line x1={midScreenX} y1={midScreenY} x2={eeScreenX} y2={eeScreenY} stroke="#2563EB" strokeWidth="4.5" strokeLinecap="round" />
                        
                        {/* End Effector Gripper Claws assemblies */}
                        <g transform={`translate(${eeScreenX}, ${eeScreenY})`}>
                          <circle cx="0" cy="0" r="5.5" fill="#EF4444" stroke="#FFFFFF" strokeWidth="1.2" />
                          <path 
                            d={isGrasping ? "M -5,3 L -1,9 M 5,3 L 1,9" : "M -7,-2 L -3,6 M 7,-2 L 3,6"} 
                            stroke="#EF4444" 
                            strokeWidth="2.5" 
                            strokeLinecap="round" 
                          />
                        </g>

                        {/* Vertical ground target laser-projection line */}
                        <line 
                          x1={eeScreenX} 
                          y1={eeScreenY} 
                          x2={eeScreenX} 
                          y2={225 - (armY * 0.4)} 
                          stroke="#F43F5E" 
                          strokeWidth="1" 
                          strokeDasharray="2 2" 
                          opacity="0.8" 
                        />
                        <circle cx={eeScreenX} cy={225 - (armY * 0.4)} r="3" fill="#F43F5E" opacity="0.9" />
                      </g>
                    );
                  })()}
                </g>
              </svg>
            </div>
          </div>

          {/* Upgraded bento document explorer with Framer-Motion switches */}
          <div className="bg-[#121420] border border-zinc-900 rounded-xl p-6 shadow-xl flex flex-col" id="academic-guide-explorer">
            
            <div className="flex flex-col gap-1 mb-5">
              <div className="flex items-center gap-2.5">
                <BookOpen className="w-4.5 h-4.5 text-blue-400" />
                <h2 className="text-xs font-bold uppercase tracking-widest font-display text-zinc-200">
                  Stage Explorer & Capstone Documentation
                </h2>
              </div>
              <p className="text-[11.5px] text-zinc-500 font-sans">
                Select corresponding system module stages to audit internal parameters and source code:
              </p>
            </div>

            {/* Stage Selector switch header buttons row */}
            <div className="flex gap-1 overflow-x-auto pb-3 border-b border-zinc-900 scrollbar-thin scrollbar-thumb-zinc-800 scrollbar-track-transparent select-none" id="guide-tabs-row">
              {(["architecture", "structure", "vision", "planner", "simulator", "dashboard", "memory", "yolo", "ros2"] as const).map((tabId, idx) => {
                const isActive = activeTab === tabId;
                const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
                
                return (
                  <button
                    key={tabId}
                    type="button"
                    onClick={() => setActiveTab(tabId)}
                    className={`py-2 px-3 text-[10.5px] font-bold tracking-wide rounded-lg font-display transition-all border shrink-0 relative ${
                      isActive 
                        ? "bg-blue-600/10 text-blue-400 border-blue-500/20 shadow-sm" 
                        : "bg-zinc-950/20 text-zinc-500 border-transparent hover:text-zinc-200 hover:bg-zinc-900"
                    }`}
                  >
                    {idx + 1}. {capitalize(tabId === "architecture" ? "Arch" : tabId === "vision" ? "Vision" : tabId === "yolo" ? "Yolo" : tabId)}
                    {isActive && (
                      <motion.div 
                        layoutId="activeTabUnderline"
                        className="absolute bottom-0 left-2 right-2 h-0.5 bg-blue-500"
                        transition={{ type: "spring", stiffness: 350, damping: 30 }}
                      />
                    )}
                  </button>
                );
              })}
            </div>

            {/* 3. Stage Documentation Switch Container */}
            <div className="pt-4 min-h-[340px]">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                  className="space-y-4"
                >
                  
                  {/* STAGE 1: System Flow flowchart diagram */}
                  {activeTab === "architecture" && (
                    <div className="space-y-4">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-xl pointer-events-none"></div>
                        <h3 className="text-xs font-bold text-blue-400 font-mono uppercase tracking-widest mb-2 flex items-center gap-1.5">
                          <Activity className="w-4 h-4" />
                          <span>STAGE 1: Modular Microservices Pipeline mapping</span>
                        </h3>
                        <p className="text-xs text-zinc-400 leading-relaxed font-sans mt-1">
                          This structured data flow trace visualizes the standard layout interaction of components for academic reviews. Each component is isolated by strict REST/JSON schemas:
                        </p>
                      </div>

                      <div className="p-4 bg-zinc-950/90 border border-zinc-900 rounded-xl overflow-x-auto">
                        <pre className="text-[10px] line-clamp-none whitespace-pre font-mono text-zinc-400 leading-relaxed max-w-full">
                          {SYSTEM_ARCHITECTURE_ASCII}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* STAGE 2: Folder Directory structural map */}
                  {activeTab === "structure" && (
                    <div className="space-y-3">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl">
                        <h3 className="text-xs font-bold text-indigo-400 font-mono uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                          <Layers className="w-4 h-4" />
                          <span>STAGE 2: System complete folder layout</span>
                        </h3>
                        <p className="text-xs text-zinc-400 font-sans leading-relaxed">
                          Clean separation of concerns, ensuring ready-to-test package files for internship submission panels:
                        </p>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                        {FOLDER_STRUCTURES.map((section, idx) => (
                          <div key={idx} className="bg-zinc-950/60 p-4 border border-zinc-900 rounded-xl flex flex-col justify-between">
                            <div>
                              <div className="flex items-center justify-between gap-1 border-b border-zinc-900/80 pb-2 mb-3">
                                <span className="font-mono text-xs font-bold text-white flex items-center gap-1">
                                  📂 {section.title}
                                </span>
                                <span className="text-[9px] px-2 py-0.5 border border-zinc-800 rounded-full font-mono text-zinc-500 uppercase">
                                  {section.badge}
                                </span>
                              </div>
                              
                              <div className="space-y-3">
                                {section.items.map((item, idy) => (
                                  <div key={idy} className="font-mono text-[10.5px]">
                                    <div className="text-zinc-200 font-bold flex items-center gap-1">
                                      <FileCode className="w-3.5 h-3.5 text-zinc-600" />
                                      <code>{item.name}</code>
                                    </div>
                                    <p className="text-zinc-500 font-sans text-xs mt-0.5 ml-4.5 font-sans leading-relaxed">
                                      {item.details}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* STAGE 3: OpenCV Webcam Pixel matrix to physical space */}
                  {activeTab === "vision" && (
                    <div className="space-y-4 font-sans">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl space-y-2">
                          <h3 className="text-xs font-bold text-emerald-400 font-display uppercase tracking-widest flex items-center gap-1.5">
                            <Eye className="w-4 h-4" />
                            <span>STAGE 3: OpenCV Masking & Calibration Transform</span>
                          </h3>
                          <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                            Transforms the pixel coordinates <strong className="text-zinc-300">(u, v)</strong> captured under Webcam streams into physical table ground space <strong className="text-zinc-300">(X, Y, Z mm)</strong> relative to calibration focal coordinates (320, 240):
                          </p>
                          <div className="bg-zinc-950 p-3.5 border border-zinc-900 rounded-lg text-[10.5px] font-mono text-emerald-400 leading-relaxed shadow-inner">
                            {`real_x_mm = (coord_u - 320) * scale_multiplier\nreal_y_mm = (240 - coord_v) * scale_multiplier\nreal_z_mm = workpiece_height # 10mm`}
                          </div>
                        </div>

                        {/* Interactive Color Mask Calibration Swatch Sandbox */}
                        <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] font-mono uppercase font-bold tracking-wider text-emerald-400">Contours calibration Sandbox</span>
                            <span className="text-[9px] font-mono text-zinc-500 bg-zinc-900 px-1.5 py-0.5 rounded">Range GUI</span>
                          </div>
                          
                          <p className="text-[11px] text-zinc-500">
                            Slide parameter thresholds to coordinate the OpenCV custom `inRange()` masking matrix dynamically below:
                          </p>

                          <div className="space-y-3.5 pt-1">
                            <div>
                              <div className="flex justify-between text-[11px] font-mono text-zinc-400 mb-1">
                                <span>Hue range Lower:</span>
                                <span className="text-white font-bold">{hsvH}</span>
                              </div>
                              <input 
                                type="range" 
                                min="0" 
                                max="180" 
                                value={hsvH} 
                                onChange={(e) => setHsvH(parseInt(e.target.value))}
                                className="w-full accent-blue-500 bg-zinc-900 h-1 rounded" 
                              />
                            </div>
                            <div>
                              <div className="flex justify-between text-[11px] font-mono text-zinc-400 mb-1">
                                <span>Saturation threshold:</span>
                                <span className="text-white font-bold">{hsvS}</span>
                              </div>
                              <input 
                                type="range" 
                                min="0" 
                                max="255" 
                                value={hsvS} 
                                onChange={(e) => setHsvS(parseInt(e.target.value))}
                                className="w-full accent-blue-500 bg-zinc-900 h-1 rounded" 
                              />
                            </div>
                            <div>
                              <div className="flex justify-between text-[11px] font-mono text-zinc-400 mb-1">
                                <span>Value illumination lower:</span>
                                <span className="text-white font-bold">{hsvV}</span>
                              </div>
                              <input 
                                type="range" 
                                min="0" 
                                max="255" 
                                value={hsvV} 
                                onChange={(e) => setHsvV(parseInt(e.target.value))}
                                className="w-full accent-blue-500 bg-zinc-900 h-1 rounded" 
                              />
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Display live code string representing computed slider params */}
                      <div className="p-4 bg-zinc-950/95 border border-zinc-900 rounded-xl font-mono text-xs text-zinc-300">
                        <span className="text-[10px] text-zinc-500 uppercase tracking-widest block mb-2 font-bold"># Computed Python OpenCV Contour Frame (cv_tracker.py)</span>
                        <pre className="text-emerald-500 text-[10px] leading-relaxed max-w-full overflow-x-auto">
{`# Mask setup generated dynamically inside calibration panel
cv_mask = cv2.inRange(hsv_frame, np.array([${hsvH}, ${hsvS}, ${hsvV}]), np.array([${hsvH + 11}, 255, 255]))
contours, _ = cv2.findContours(cv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 300:
        coord_m = cv2.moments(c)
        cx = int(coord_m["m10"] / coord_m["m00"])
        cy = int(coord_m["m01"] / coord_m["m00"])
        # Coordinates translation relative to resolution focal lengths
        real_x = (cx - 320) * 1.25`}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* STAGE 4: Gemini Cognitive VLA Plan Schema comparison */}
                  {activeTab === "planner" && (
                    <div className="space-y-4 font-sans">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl">
                        <h3 className="text-xs font-bold text-blue-400 font-mono uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                          <Sparkles className="w-4 h-4" />
                          <span>STAGE 4: Cognitive Brain schema payload formulation</span>
                        </h3>
                        <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                          Cognitive VLA clients formulate granular trajectories using the prompt structure database parameters. Gemini is configured with strict JSON schemas to ensure zero conversational overhead:
                        </p>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                        <div className="bg-zinc-950/90 border border-zinc-900 p-4 rounded-xl space-y-2">
                          <span className="text-blue-400 font-bold block border-b border-zinc-900 pb-1 text-[10px] uppercase">1. Client API JSON target payload (POST)</span>
                          <pre className="text-[10px] text-blue-300 overflow-x-auto whitespace-pre leading-relaxed">
                            {PLANNER_INPUT_PAYLOAD}
                          </pre>
                        </div>
                        <div className="bg-zinc-950/90 border border-zinc-900 p-4 rounded-xl space-y-2">
                          <span className="text-emerald-400 font-bold block border-b border-zinc-900 pb-1 text-[10px] uppercase">2. Synthesized Gemini JSON output plan</span>
                          <pre className="text-[10px] text-emerald-400 overflow-x-auto whitespace-pre leading-relaxed">
                            {PLANNER_RESPONSE_PLAN}
                          </pre>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* STAGE 5: Bi-directional Kinematics Live Calculator & math logic */}
                  {activeTab === "simulator" && (
                    <div className="space-y-4 font-sans">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl pointer-events-none"></div>
                        <h3 className="text-xs font-bold text-indigo-400 font-mono uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                          <SlidersHorizontal className="w-4 h-4" />
                          <span>STAGE 5: Analytical Inverse Kinematics Live math Playground</span>
                        </h3>
                        <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                          Drag parameters in this calculator to compute trigonometry parameters and **physically update the simulator SVG robotic arm coordinates live**! Joint lengths are defined as: L1 (Shoulder)=80mm, L2 (Upper link)=100mm, L3 (Forearm link)=90mm.
                        </p>
                      </div>

                      {/* Bi-directional math playground card */}
                      <div className="p-5 bg-zinc-950 border border-zinc-900 rounded-xl space-y-4">
                        <div className="flex items-center gap-2 border-b border-zinc-900 pb-2.5">
                          <Gauge className="w-4.5 h-4.5 text-indigo-400" />
                          <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-300 font-bold block">Live joints coordinate mapping slider panels</span>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="space-y-1 bg-zinc-900/40 p-3 border border-zinc-900/60 rounded-xl">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="text-zinc-400">Position X (mm):</span>
                              <span className="text-white font-bold">{armX}mm</span>
                            </div>
                            <input 
                              type="range" 
                              min="-140" 
                              max="140" 
                              value={armX} 
                              onChange={(e) => updateInverseKinematics(parseInt(e.target.value), armY, armZ)}
                              className="w-full accent-blue-500 bg-zinc-950/90 h-1 rounded pointer" 
                            />
                            <span className="text-[9.5px] font-mono text-zinc-650 block text-right mt-1 font-bold">Table Width Limits [-140, 140]</span>
                          </div>

                          <div className="space-y-1 bg-zinc-900/40 p-3 border border-zinc-900/60 rounded-xl">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="text-zinc-400">Position Y (mm):</span>
                              <span className="text-white font-bold">{armY}mm</span>
                            </div>
                            <input 
                              type="range" 
                              min="100" 
                              max="210" 
                              value={armY} 
                              onChange={(e) => updateInverseKinematics(armX, parseInt(e.target.value), armZ)}
                              className="w-full accent-emerald-500 bg-zinc-950/90 h-1 rounded pointer" 
                            />
                            <span className="text-[9.5px] font-mono text-zinc-650 block text-right mt-1 font-bold">Table Reach Limits [100, 210]</span>
                          </div>

                          <div className="space-y-1 bg-zinc-900/40 p-3 border border-zinc-900/60 rounded-xl">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="text-zinc-400">Height Z (mm):</span>
                              <span className="text-white font-bold">{armZ}mm</span>
                            </div>
                            <input 
                              type="range" 
                              min="0" 
                              max="140" 
                              value={armZ} 
                              onChange={(e) => updateInverseKinematics(armX, armY, parseInt(e.target.value))}
                              className="w-full accent-amber-500 bg-zinc-950/90 h-1 rounded pointer" 
                            />
                            <span className="text-[9.5px] font-mono text-zinc-650 block text-right mt-1 font-bold">Table Height Limits [0, 140]</span>
                          </div>
                        </div>

                        {/* Computed degree results */}
                        <div className="p-3 bg-zinc-900/60 border border-zinc-900/80 rounded-xl grid grid-cols-3 gap-3 text-center text-xs font-mono shadow-inner">
                          <div>
                            <span className="text-zinc-500 block mb-0.5 text-[10px]">theta1 (Base Azimuth)</span>
                            <span className="text-blue-400 font-bold text-sm block">{theta1}°</span>
                          </div>
                          <div>
                            <span className="text-zinc-500 block mb-0.5 text-[10px]">theta2 (Shoulder lift)</span>
                            <span className="text-emerald-400 font-bold text-sm block">{theta2}°</span>
                          </div>
                          <div>
                            <span className="text-zinc-500 block mb-0.5 text-[10px]">theta3 (Elbow flexion)</span>
                            <span className="text-amber-400 font-bold text-sm block">{theta3}°</span>
                          </div>
                        </div>

                      </div>
                    </div>
                  )}

                  {/* STAGE 6: Streamlit Operator CLI Console Code */}
                  {activeTab === "dashboard" && (
                    <div className="space-y-4">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl">
                        <h3 className="text-xs font-bold text-indigo-400 font-mono uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                          <Smartphone className="w-4 h-4" />
                          <span>STAGE 6: Python operator GUI Dashboard</span>
                        </h3>
                        <p className="text-xs text-zinc-400 font-sans leading-relaxed">
                          The computer workspace executes an asynchronous operator feed, connecting to backend FastAPI routes on Port 3000 over default TCP headers:
                        </p>
                      </div>

                      <div className="bg-zinc-950/95 border border-zinc-900 p-4 rounded-xl font-mono text-xs">
                        <span className="text-[10px] text-zinc-500 uppercase tracking-widest block mb-2 font-bold select-none"># Dashboard core script (dashboard_app.py)</span>
                        <pre className="text-zinc-300 leading-relaxed overflow-x-auto text-[10px]">
                          {STREAMLIT_APP_CODE}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* STAGE 7: SQLite Memory schemas schema mapping */}
                  {activeTab === "memory" && (
                    <div className="space-y-4">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl">
                        <h3 className="text-xs font-bold text-emerald-400 font-mono uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                          <Database className="w-4 h-4" />
                          <span>STAGE 7: SQLite persistent relational DB schemas DDL</span>
                        </h3>
                        <p className="text-xs text-zinc-400 font-sans leading-relaxed">
                          Maintains physical table states and system execution run timings to secure outstanding evaluation metrics with presentation panels:
                        </p>
                      </div>

                      <div className="bg-zinc-950/95 border border-zinc-900 p-4 rounded-xl font-mono text-xs text-zinc-400">
                        <span className="text-[10px] text-zinc-500 uppercase tracking-widest block mb-1.5 font-bold select-none">-- Relational Tables creation statements (history_db.py)</span>
                        <pre className="leading-relaxed overflow-x-auto text-[10px] text-zinc-300">
                          {SQL_SCHEMA_STATEMENTS}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* STAGE 8: YOLO Deep Learning Centroid predictions */}
                  {activeTab === "yolo" && (
                    <div className="space-y-4">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl space-y-2">
                        <h3 className="text-xs font-bold text-blue-400 font-mono uppercase tracking-widest flex items-center gap-1.5">
                          <Code2 className="w-4 h-4" />
                          <span>STAGE 8: YOLOv8 Computer Vision bounding centration parameters</span>
                        </h3>
                        <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                          Employs custom convolutional parameters (YOLOv8 framework model) to securely track centroids under messy objects configurations:
                        </p>
                        
                        <div className="bg-zinc-900/30 p-3.5 border border-zinc-900/80 rounded-xl flex items-start gap-2.5 text-xs">
                          <Info className="w-5 h-5 text-zinc-400 shrink-0 mt-0.5" />
                          <div className="leading-relaxed text-zinc-450 font-sans text-xs">
                            <strong>Advantages Over OpenCV HSV Threshold Masks:</strong> Light spectrum insensitivity, robust tracking under partial item occlusions, and automated category labeling markers.
                          </div>
                        </div>
                      </div>

                      <div className="bg-zinc-950/95 border border-zinc-900 p-4 rounded-xl font-mono text-xs">
                        <span className="text-[10px] text-zinc-400 uppercase tracking-widest block mb-2 font-bold select-none"># ONNX Tabletop predictions coordinate solver (yolo_track.py)</span>
                        <pre className="text-zinc-300 overflow-x-auto leading-relaxed text-[10px]">
                          {YOLO_CODE_SNIPPET}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* STAGE 9: ROS 2 Controller Publisher Node snippet */}
                  {activeTab === "ros2" && (
                    <div className="space-y-4">
                      <div className="p-4 bg-zinc-950/80 border border-zinc-900 rounded-xl">
                        <h3 className="text-xs font-bold text-amber-500 font-mono uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                          <Terminal className="w-4 h-4" />
                          <span>STAGE 9: Autonomous ROS 2 Humble controller publisher node</span>
                        </h3>
                        <p className="text-xs text-zinc-400 font-sans leading-relaxed">
                          Standard Python publisher node converting target joint matrices into Radian-transition arrays, posted onto `/vla_arm_controller/joint_trajectory`:
                        </p>
                      </div>

                      <div className="bg-zinc-950/95 border border-zinc-900 p-4 rounded-xl font-mono text-xs">
                        <span className="text-[10px] text-zinc-500 uppercase tracking-widest block mb-2 font-bold select-none"># ROS 2 Humblest joints Publisher Code snippet (controller_pub.py)</span>
                        <pre className="text-zinc-300 overflow-x-auto leading-relaxed text-[10.5px]">
                          {ROS2_NODE_CODE}
                        </pre>
                      </div>
                    </div>
                  )}

                </motion.div>
              </AnimatePresence>
            </div>

          </div>

        </section>

      </main>

      {/* 4. Footer: Clean design credit */}
      <footer className="border-t border-zinc-900 bg-[#07090F] text-zinc-500 text-xs px-6 py-8 text-center max-w-full">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 font-sans text-zinc-500">
          <p className="text-[11px] font-mono tracking-tight text-left">
            AI Robotics VLA System © 2026 • CSE Capstone Engineering Platform for Internship evaluations
          </p>
          <div className="flex items-center gap-1.5 text-[10px]">
            <Lock className="w-3.5 h-3.5 text-zinc-600" />
            <span className="text-zinc-600 font-mono select-none uppercase">Secured sandboxed environment on Cloud Run</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
