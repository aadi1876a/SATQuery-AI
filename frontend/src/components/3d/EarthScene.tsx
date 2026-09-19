import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { Earth } from './Earth';
import { Satellite } from './Satellite';

interface EarthSceneProps {
  isAnalyzing: boolean;
}

export const EarthScene: React.FC<EarthSceneProps> = ({ isAnalyzing }) => {
  return (
    <div className="w-full h-full absolute inset-0 -z-10">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
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
          <group position={isAnalyzing ? [0, 0, -2] : [0, 0, 0]}>
            <Earth />
            <Satellite radius={2.6} speed={isAnalyzing ? 0.2 : 0.8} />
            <Satellite radius={3.2} speed={isAnalyzing ? 0.3 : 0.5} />
          </group>
        </Suspense>

        <OrbitControls 
          enablePan={false}
          enableZoom={!isAnalyzing}
          minDistance={3}
          maxDistance={10}
          autoRotate={!isAnalyzing}
          autoRotateSpeed={0.5}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
};
