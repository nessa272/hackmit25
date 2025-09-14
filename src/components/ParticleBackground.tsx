"use client";
import { useEffect, useRef, useState, useCallback } from "react";

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  targetX?: number;
  targetY?: number;
  size: number;
  opacity: number;
  color: string;
}

export default function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const [isInfinityMode, setIsInfinityMode] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    startTimeRef.current = performance.now();
  }, []);

  useEffect(() => {
    const updateDimensions = () => {
      setDimensions({ width: window.innerWidth, height: window.innerHeight });
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  const createParticles = useCallback(() => {
    if (dimensions.width === 0 || dimensions.height === 0) return;

    const particleCount = Math.min(80, Math.floor(dimensions.width * dimensions.height / 15000));
    const newParticles: Particle[] = [];
    const colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe", "#ffffff"];

    for (let i = 0; i < particleCount; i++) {
      newParticles.push({
        id: i,
        x: Math.random() * dimensions.width,
        y: Math.random() * dimensions.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
        opacity: Math.random() * 0.8 + 0.2,
        color: colors[Math.floor(Math.random() * colors.length)]
      });
    }

    particlesRef.current = newParticles;
  }, [dimensions]);

  const getInfinityPosition = useCallback((t: number, centerX: number, centerY: number, scale: number = 80) => {
    const x = centerX + (scale * Math.cos(t)) / (1 + Math.sin(t) ** 2);
    const y = centerY + (scale * Math.sin(t) * Math.cos(t)) / (1 + Math.sin(t) ** 2);
    return { x, y };
  }, []);

  const updateParticles = useCallback(() => {
    particlesRef.current = particlesRef.current.map((particle, index) => {
      let newX = particle.x;
      let newY = particle.y;
      let newVx = particle.vx;
      let newVy = particle.vy;

      if (isInfinityMode) {
        const centerX = dimensions.width / 2;
        const centerY = dimensions.height / 2 - 50;

        // Create a smoother infinity curve by spacing particles more evenly
        const t = (index / particlesRef.current.length) * 4 * Math.PI + (performance.now() - startTimeRef.current) * 0.0003;
        const infinityPos = getInfinityPosition(t, centerX, centerY, Math.min(200, dimensions.width * 0.2));

        const dx = infinityPos.x - particle.x;
        const dy = infinityPos.y - particle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Stronger attraction force for seamless movement
        if (distance > 3) {
          newVx = dx * 0.08;
          newVy = dy * 0.08;
        } else {
          // Once close, follow the curve smoothly
          
          const nextT = t + 0.1;
          const nextPos = getInfinityPosition(nextT, centerX, centerY, Math.min(200, dimensions.width * 0.2));
          newVx = (nextPos.x - particle.x) * 0.1;
          newVy = (nextPos.y - particle.y) * 0.1;
        }
      } else {
        if (particle.targetX !== undefined && particle.targetY !== undefined) {
          const dx = particle.targetX - particle.x;
          const dy = particle.targetY - particle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance > 5) {
            newVx = dx * 0.05;
            newVy = dy * 0.05;
          } else {
            newVx = (Math.random() - 0.5) * 0.5;
            newVy = (Math.random() - 0.5) * 0.5;
            particle.targetX = undefined;
            particle.targetY = undefined;
          }
        } else {
          // Normal floating behavior
          newVx += (Math.random() - 0.5) * 0.02;
          newVy += (Math.random() - 0.5) * 0.02;

          // Limit velocity
          const speed = Math.sqrt(newVx * newVx + newVy * newVy);
          if (speed > 1) {
            newVx = (newVx / speed) * 1;
            newVy = (newVy / speed) * 1;
          }
        }
      }

      newX += newVx;
      newY += newVy;

      // Wrap around screen edges
      if (newX < 0) newX = dimensions.width;
      if (newX > dimensions.width) newX = 0;
      if (newY < 0) newY = dimensions.height;
      if (newY > dimensions.height) newY = 0;

      return {
        ...particle,
        x: newX,
        y: newY,
        vx: newVx,
        vy: newVy
      };
    });
  }, [isInfinityMode, dimensions, getInfinityPosition]);

  const drawParticles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    particlesRef.current.forEach(particle => {
      ctx.save();
      ctx.globalAlpha = particle.opacity;
      ctx.fillStyle = particle.color;

      // Minimal effects for performance
      if (isInfinityMode) {
        ctx.shadowColor = "#3b82f6";
        ctx.shadowBlur = particle.size;
      }

      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    });

    // No connection lines - just particles forming the infinity shape
  }, [dimensions, isInfinityMode, getInfinityPosition]);

  useEffect(() => {
    createParticles();
  }, [createParticles]);

  useEffect(() => {
    if (particlesRef.current.length === 0) return;

    const animate = (currentTime: number) => {
      // Limit to ~60 FPS for better performance
      if (currentTime - lastFrameTimeRef.current >= 16) {
        updateParticles();
        drawParticles();
        lastFrameTimeRef.current = currentTime;
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [updateParticles, drawParticles]);

  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    console.log("Canvas clicked!", {
      x: e.clientX,
      y: e.clientY,
      isInfinityMode,
      particleCount: particlesRef.current.length,
      dimensions
    });

    // Create particles if they don't exist yet
    if (particlesRef.current.length === 0) {
      console.log("Creating particles on click");
      createParticles();
      return;
    }

    if (!isInfinityMode) {
      console.log("Starting infinity mode");
      setIsInfinityMode(true);
      setTimeout(() => {
        console.log("Ending infinity mode");
        setIsInfinityMode(false);
        particlesRef.current = particlesRef.current.map(particle => ({
          ...particle,
          targetX: Math.random() * dimensions.width,
          targetY: Math.random() * dimensions.height
        }));
      }, 5000);
    }
  }, [isInfinityMode, dimensions, createParticles]);

  if (!isClient) {
    return (
      <div className="fixed inset-0 w-full h-full z-0">
        <div
          className="absolute inset-0 w-full h-full"
          style={{
            background: "linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%)"
          }}
        />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 w-full h-full z-0">
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        onClick={handleCanvasClick}
        className="absolute inset-0 w-full h-full cursor-pointer"
        style={{
          background: "linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%)",
          pointerEvents: "auto"
        }}
      />
    </div>
  );
}