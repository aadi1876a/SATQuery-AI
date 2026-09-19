import React, { Suspense, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { Earth } from './Earth';
import { Satellite } from './Satellite';

// ─── Inner group that lerps its own position (avoids React re-render on every
//     frame; position is mutated directly on the mesh ref inside useFrame). ─────
const SceneGroup: React.FC<{ isAnalyzing: boolean }> = ({ isAnalyzing }) => {
  const groupRef = useRef<THREE.Group>(null);

  // Stable target positions — allocated once, not on every render.
  const posAnalyzing = useMemo(() => new THREE.Vector3(0, 0, -2), []);
  const posIdle      = useMemo(() => new THREE.Vector3(0, 0,  0), []);
  const targetPos    = isAnalyzing ? posAnalyzing : posIdle;

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    // Smooth lerp toward the target — 3 units/s effective speed gives a
    // ~700-900 ms cubic-ease-like feel (lerp factor chosen so the first
    // frame moves fastest and it slows as it nears the target).
    groupRef.current.position.lerp(targetPos, 1 - Math.pow(0.01, delta * 4));
  });

  return (
    <group ref={groupRef}>
      <Earth />
      {/* Two satellites on different radii and orbital speeds.
          speed in rad/s: 0.12 → ~52 s/rev, 0.10 → ~63 s/rev.
          During analysis, slow down to 0.07 (~90 s) for a calmer feel. */}
      <Satellite radius={2.6} speed={isAnalyzing ? 0.07 : 0.12} />
      <Satellite radius={3.2} speed={isAnalyzing ? 0.055 : 0.10} />
    </group>
  );
};

interface EarthSceneProps {
  isAnalyzing: boolean;
}

export const EarthScene: React.FC<EarthSceneProps> = ({ isAnalyzing }) => {
  return (
    <div className="w-full h-full absolute inset-0 -z-10">
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        // Cap device pixel ratio to 2 to avoid GPU overload on HiDPI screens.
        // performance.min=0.5 lets R3F lower dpr under load rather than dropping frames.
        dpr={[1, 2]}
        performance={{ min: 0.5 }}
      >
        <color attach="background" args={['#02050a']} />

        {/* Subtle space background */}
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />

        <ambientLight intensity={isAnalyzing ? 0.2 : 0.4} />
        <directionalLight
          position={[5, 3, 5]}
          intensity={isAnalyzing ? 1.5 : 2}
          color={isAnalyzing ? '#00d4ff' : '#ffffff'}
        />
        <directionalLight
          position={[-5, -3, -5]}
          intensity={0.2}
          color="#0088cc"
        />

        <Suspense fallback={null}>
          <SceneGroup isAnalyzing={isAnalyzing} />
        </Suspense>

        <OrbitControls
          enablePan={false}
          enableZoom={!isAnalyzing}
          minDistance={3}
          maxDistance={10}
          // Disable OrbitControls autoRotate — Earth already rotates via its own
          // delta-based useFrame. Having both active causes double-rotation and
          // the OrbitControls speed is frame-rate dependent, not delta-based.
          autoRotate={false}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
};
