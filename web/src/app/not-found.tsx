import Link from "next/link";
import { PageHeader } from "@/components/ui";

export default function NotFound() {
  return (
    <PageHeader title="Page not found" intro={<p>That page does not exist. Try the <Link href="/districts">districts</Link>, <Link href="/municipalities">municipalities</Link> or the <Link href="/">home page</Link>.</p>} />
  );
}
