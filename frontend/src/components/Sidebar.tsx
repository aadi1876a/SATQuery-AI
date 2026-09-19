import React from 'react';
import { TaskType } from '../types/api';
import { LayoutDashboard, MessageSquare, Tag, Crosshair, History, SplitSquareHorizontal, Layers } from 'lucide-react';
import './Sidebar.css';

interface SidebarProps {
  activeTask: TaskType | 'dashboard';
  setActiveTask: (task: TaskType | 'dashboard') => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeTask, setActiveTask }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'vqa', label: 'Visual QA', icon: MessageSquare },
    { id: 'captioning', label: 'Captioning', icon: Tag },
    { id: 'grounding', label: 'Grounding', icon: Crosshair },
    { id: 'change_vqa', label: 'Change Analysis', icon: History, disabled: true },
    { id: 'fusion_analysis', label: 'SAR+Optical Fusion', icon: SplitSquareHorizontal, disabled: true },
  ];

  return (
    <aside className="sidebar-container">
      <div className="sidebar-section">
        <h3 className="sidebar-title flex items-center gap-2">
          <Layers size={16} /> Analysis Modes
        </h3>
        <nav className="sidebar-nav">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`sidebar-item ${activeTask === item.id ? 'active' : ''} ${item.disabled ? 'disabled' : ''}`}
              onClick={() => !item.disabled && setActiveTask(item.id as TaskType | 'dashboard')}
              disabled={item.disabled}
              title={item.disabled ? 'Coming from backend' : ''}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
              {item.disabled && <span className="badge-coming">WIP</span>}
            </button>
          ))}
        </nav>
      </div>
    </aside>
  );
};

export default Sidebar;
