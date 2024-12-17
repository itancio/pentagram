import { AppSidebar } from "@/components/AppSidebar";
// import { NavActions } from "@/components/nav-actions"
import { SidebarProvider } from "@/components/ui/sidebar";

export default function Page() {
  return (
    <SidebarProvider>
      <AppSidebar />
    </SidebarProvider>
  );
}
